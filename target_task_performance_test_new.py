import os

import torch
import torch.nn as nn
import h5py
import torch.optim as optim
from tqdm import tqdm
from data.celeba.celeba_dataset import CelebA_DATASET
from utils.utils import *
from utils.setup_md import setup_datasets, select_model
from torch.utils.data import Dataset, DataLoader

# prop_ids = [15, 20, 31, 35, 39]
prop_ids = [15, 20, 31, 39]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-wandb',
                        action='store_true',
                        default=False,
                        help='Use WandB?')

    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')

    parser.add_argument('--wd',
                        type=float,
                        default=1e-4,
                        help='weight decay')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='momentum')

    parser.add_argument('--finetune-epochs',
                        type=int,
                        default=20,
                        help='finetune-epochs')

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

    parser.add_argument('--num-seeds',
                        type=int,
                        default=5)

    parser.add_argument('--cuda-no',
                        type=int,
                        default=1)

    parser.add_argument('--linear-probing',
                        type=int,
                        default=0)

    parser.add_argument('--seed',
                        type=int,
                        default=42)

    parser.add_argument('--interval',
                        type=int,
                        default=50)

    parser.add_argument('--note',
                        type=str,
                        default="")

    return parser.parse_args()

args = parse_args()
cuda_no = args.cuda_no
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_no}"
lr = args.lr
wd = args.wd
momentum = args.momentum
finetune_epochs = args.finetune_epochs
num_seeds = args.num_seeds
dir_name = args.dir_name
linear_probing = args.linear_probing
target_PID = args.target_PID
target_PID_index = prop_ids.index(target_PID)
num_samples = args.auxiliary_train_samples
interval = args.interval
def test(model=None, data_loader=None):
    model.eval()
    total_right = 0
    total_samples = 0
    with torch.no_grad():
        for step, (x, multi_labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            logits = model(x)
            y = multi_labels[:, target_PID_index]  # target task label
            preds = torch.argmax(logits, dim=-1)
            total_right += torch.sum(torch.eq(preds, y))
            total_samples += len(y)
        acc = float(total_right) / total_samples
    torch.cuda.empty_cache()
    return acc

def finetune_test(model_path=None):
    model = torch.load(model_path)
    num_features = model.decoder.in_features
    model.decoder = nn.Linear(num_features, 2)

    # initialization
    nn.init.xavier_normal_(model.decoder.weight)
    nn.init.zeros_(model.decoder.bias)

    model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    if linear_probing:
        optimizer = optim.SGD(params=model.decoder.parameters(),
                              lr=lr,
                              weight_decay=wd,
                              momentum=momentum)
    else:
        optimizer = optim.SGD(params=model.parameters(),
                              lr=lr,
                              weight_decay=wd,
                              momentum=momentum)
    for _ in tqdm(range(finetune_epochs)):
        model.train()
        for step, (x, multi_labels) in enumerate(auxiliary_train_loader):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            # if model.probabilistic:
            #     z, (z_mu, z_sigma) = model.featurize(x)
            # else:
            #     z_mu = model.featurize(x)
            z_params = model.encoder(x)
            z_mu = z_params[:, :model.z_dim]
            logits = model.decoder(z_mu)
            target_labels = multi_labels[:, target_PID_index]  # target task label
            loss = loss_fn(logits, target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_acc = test(model=model, data_loader=test_loader)
    print(f"model_path {model_path}, test acc: {test_acc:.4f}")
    return test_acc

def transfer_test(dir_name):
    dir_path = os.path.join('./fl_models/', dir_name)
    names = os.listdir(dir_path)
    # rounds = sorted([int(item.split('-')[0]) for item in names], reverse=True)
    rounds = sorted([int(item.split('-')[0]) for item in names])
    for r in rounds:
        if r % interval != 0:
            continue
        model_path = os.path.join(dir_path, f'{r}-model.pt')
        # print(model_path)
        results = []
        for seed in range(num_seeds):
            setup_seed(seed)
            test_acc = finetune_test(model_path=model_path)
            results.append(test_acc)
        if args.use_wandb:
            wandb.log({
                'round': r,
                'mean': np.mean(results),
                'std': np.std(results, ddof=1)
            })
        else:
            print("round", r,
                  "mean", "{:.4f}".format(np.mean(results)),
                  "std", "{:.4f}".format(np.std(results, ddof=1)))

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