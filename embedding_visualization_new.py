from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.utils import *
from utils.setup_md import setup_datasets, select_model
from torchvision import models
from data.celeba.celeba_dataset_bakkk import CelebA_DATASET
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# prop_ids = [15, 20, 31, 35, 39]
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

    parser.add_argument('--cuda-no',
                        type=int,
                        default=1)

    parser.add_argument('--seed',
                        type=int,
                        default=42)

    parser.add_argument('--interval',
                        type=int,
                        default=50)

    parser.add_argument('--K',
                        type=int,
                        default=5)

    parser.add_argument('--note',
                        type=str,
                        default="")

    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda_no}"
if args.use_wandb:
    import wandb

    wandb.init(project="VL", entity="tdye24")
    wandb.watch_called = False
    config = wandb.config
    config.update(args)
else:
    config = args

train_loaders, test_loader = setup_datasets(config=config)
# 获取数据集
all_train_img_names = []
for loader in train_loaders:
    all_train_img_names.extend(loader.dataset.image_names)

all_train_d = CelebA_DATASET(image_names=all_train_img_names,
                        attr_transform=lambda x: torch.tensor(x[config.target_PID]).long())
all_train_loader = DataLoader(dataset=all_train_d, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


all_test_img_names = test_loader.dataset.image_names
all_test_d = CelebA_DATASET(image_names=all_test_img_names,
                        attr_transform=lambda x: torch.tensor(x[config.target_PID]).long())
all_test_loader = DataLoader(dataset=all_test_d, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


setup_seed(rs=42)

dir_path = os.path.join('./fl_models/', config.dir_name)
names = os.listdir(dir_path)
rounds = sorted([int(item.split('-')[0]) for item in names])
for r in rounds:
    if r % config.interval != 0:
        continue
    path = f'./fl_models/{config.dir_name}/{r}-model.pt'
    model = torch.load(path)
    model.cuda()
    model.eval()

    train_embeddings = []
    train_labels = []
    with torch.no_grad():
        for step, (inputs, labels) in tqdm(enumerate(all_train_loader)):
            inputs = inputs.cuda()
            z_params = model.encoder(inputs)
            z_mu = z_params[:, :model.z_dim]
            batch_labels = np.array(labels)
            train_embeddings.append(z_mu.cpu().detach().numpy())
            train_labels.append(batch_labels)

    test_embeddings = []
    test_labels = []
    with torch.no_grad():
        for step, (inputs, labels) in tqdm(enumerate(all_test_loader)):
            inputs = inputs.cuda()
            z_params = model.encoder(inputs)
            z_mu = z_params[:, :model.z_dim]
            batch_labels = np.array(labels)
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

    # 预测训练集的标签
    y_train_pred = knn.predict(X_train_tensor)
    train_accuracy = accuracy_score(y_train_tensor, y_train_pred)
    # 预测测试集的标签
    y_test_pred = knn.predict(X_test_tensor)
    test_accuracy = accuracy_score(y_test_tensor, y_test_pred)
    print(f'Round: {r}')
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')
    if config.use_wandb:
        wandb.log({
            'round': r,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        })
