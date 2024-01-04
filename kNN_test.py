import os

import torch
import torch.nn as nn
import h5py
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from data.celeba.celeba_dataset import CelebA_DATASET
from utils.utils import *
from utils.setup_md import setup_datasets, select_model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

prop_ids = [15, 20, 31, 39]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-wandb',
                        action='store_true',
                        default=False,
                        help='Use WandB?')

    parser.add_argument('--dir-name',
                        type=str,
                        default=None,
                        help='a specific model path or dir')

    parser.add_argument('--num-users',
                        type=int,
                        default=4)

    parser.add_argument('--target-PID',
                        type=int,
                        default=31)

    parser.add_argument('--batch-size',
                        type=int,
                        default=16)

    parser.add_argument('--auxiliary-train-samples',
                        type=int,
                        default=100)

    parser.add_argument('--cuda-no',
                        type=int,
                        default=1)

    parser.add_argument('--interval',
                        type=int,
                        default=50)

    parser.add_argument('--note',
                        type=str,
                        default="")

    parser.add_argument('--K',
                        type=int,
                        default=5)

    return parser.parse_args()

args = parse_args()
cuda_no = args.cuda_no
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_no}"
dir_name = args.dir_name
target_PID = args.target_PID
target_PID_index = prop_ids.index(target_PID)
num_samples = args.auxiliary_train_samples
interval = args.interval

def kNN_testing(model_path=None):
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    train_embeddings = []
    train_labels = []
    with torch.no_grad():
        for step, (inputs, labels) in tqdm(enumerate(auxiliary_train_loader)):
            inputs = inputs.cuda()
            z_params = model.encoder(inputs)
            z_mu = z_params[:, :model.z_dim]
            batch_labels = np.array(labels[:, target_PID_index])
            train_embeddings.append(z_mu.cpu().detach().numpy())
            train_labels.append(batch_labels)

    test_embeddings = []
    test_labels = []
    with torch.no_grad():
        for step, (inputs, labels) in tqdm(enumerate(test_loader)):
            inputs = inputs.cuda()
            z_params = model.encoder(inputs)
            z_mu = z_params[:, :model.z_dim]
            batch_labels = np.array(labels[:, target_PID_index])
            test_embeddings.append(z_mu.cpu().detach().numpy())
            test_labels.append(batch_labels)

    X_train = np.concatenate(train_embeddings)
    y_train = np.concatenate(train_labels)

    X_test = np.concatenate(test_embeddings)
    y_test = np.concatenate(test_labels)

    # 转换为 PyTorch 的 Tensor
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).long()

    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).long()

    print(f"Len of train={len(y_train_tensor)}")
    print(f"Len of test={len(y_test_tensor)}")

    # 定义 kNN 分类器并拟合训练数据
    knn = KNeighborsClassifier(n_neighbors=config.K)  # 这里选择了 k=3
    knn.fit(X_train, y_train)

    # 预测测试集的标签
    y_test_pred = knn.predict(X_test_tensor)
    test_accuracy = accuracy_score(y_test_tensor, y_test_pred)
    return test_accuracy

def transfer_test(dir_name):
    dir_path = os.path.join('./fl_models/', dir_name)
    names = os.listdir(dir_path)
    # rounds = sorted([int(item.split('-')[0]) for item in names], reverse=True)
    rounds = sorted([int(item.split('-')[0]) for item in names])
    for r in rounds:
        if r % interval != 0:
            continue
        model_path = os.path.join(dir_path, f'{r}-model.pt')
        test_acc = kNN_testing(model_path=model_path)
        if args.use_wandb:
            wandb.log({
                'round': r,
                'test_acc': test_acc,
            })
        else:
            print("round", r,
                  "test_acc", "{:.4f}".format(test_acc))

if args.use_wandb:
    import wandb

    wandb.init(project="VL", entity="tdye24")
    wandb.watch_called = False
    config = wandb.config
    config.update(args)
else:
    config = args
all_data_path = os.path.dirname('./data/celeba/all/')
all_clients, all_dataset = read_dir(all_data_path)
all_img_names = []
for c in all_clients:
    all_img_names.extend(all_dataset[c]['x'])
print(f"Total samples:{len(all_img_names)}.")

with open('/home/tdye/VL/data/celeba/training_img_names.json', 'r') as f:
    training_img_names = json.load(f)
with open('/home/tdye/VL/data/celeba/test_img_names.json', 'r') as f:
    test_img_names = json.load(f)
training_test_img_names = training_img_names + test_img_names
print(f"Training_test samples:{len(training_test_img_names)}.")
# auxiliary_train_img_names = [item for item in all_img_names if item not in training_test_img_names]
candidates = list(set(all_img_names) - set(training_test_img_names))
print(f"Candidates of auxiliary training samples: {len(candidates)}.")
train_loaders, test_loader = setup_datasets(config=config)
setup_seed(42)

random.shuffle(candidates)
num_prop = 0
num_nonprop = 0
hdf5_path = '/home/tdye/VL/data/celeba/data_with_labels.h5'
file = h5py.File(hdf5_path, 'r')
auxiliary_train_img_names = []
for img_name in tqdm(candidates):
    attrs = file[img_name].attrs['attributes']
    if attrs[target_PID] == 1:
        if num_prop < (num_samples // 2):
            auxiliary_train_img_names.append(img_name)
            num_prop += 1
    else:
        if num_nonprop < (num_samples // 2):
            auxiliary_train_img_names.append(img_name)
            num_nonprop += 1
    if num_prop == (num_samples // 2) and num_nonprop == (num_samples // 2):
        break

auxiliary_train_d = CelebA_DATASET(image_names=auxiliary_train_img_names,
                            attr_transform=lambda x: torch.tensor([x[prop_ids[0]],
                                                                   x[prop_ids[1]],
                                                                   x[prop_ids[2]],
                                                                   x[prop_ids[3]]]).long())
auxiliary_train_loader = DataLoader(dataset=auxiliary_train_d, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

transfer_test(dir_name=dir_name)