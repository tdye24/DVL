import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from utils.setup_md import setup_datasets, select_model
from torchvision import models

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-wandb',
                        action='store_true',
                        default=False,
                        help='Use WandB?')

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        choices=['leaf', 'resnet18'],
                        default='leaf')

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

    parser.add_argument('--path',
                        type=str,
                        default=None,
                        help='a specific model path or dir')

    parser.add_argument('--main-task',
                        type=str,
                        default='PC')

    parser.add_argument('--main-PID',
                        type=int,
                        default=20)

    parser.add_argument('--target-PID',
                        type=int,
                        default=31)

    parser.add_argument('--num-users',
                        type=int,
                        default=4)

    parser.add_argument('--batch-size',
                        type=int,
                        default=16)

    parser.add_argument('--auxiliary-train-samples',
                        type=int,
                        default=100)

    parser.add_argument('--num-seeds',
                        type=int,
                        default=5)

    parser.add_argument('--light',
                        type=int,
                        default=1)

    parser.add_argument('--tolerance',
                        type=int,
                        default=20)

    parser.add_argument('--cuda-no',
                        type=int,
                        default=1)

    parser.add_argument('--probabilistic',
                        help='probabilistic, ture (1) or false (0)',
                        type=int,
                        default=1)

    parser.add_argument('--z-dim',
                        help='z-dim',
                        type=int,
                        default=512)

    return parser.parse_args()

args = parse_args()
cuda_no = args.cuda_no
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_no}"
lr = args.lr
wd = args.wd
momentum = args.momentum
finetune_epochs = args.finetune_epochs
tolerance = args.tolerance
num_seeds = args.num_seeds
light_finetune = args.light
path = args.path
if not light_finetune:
    finetune_epochs = 2000
def test(model=None, data_loader=None):
    model.eval()
    total_right = 0
    total_samples = 0
    with torch.no_grad():
        for step, (x, multi_labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            logits = model(x)
            y = multi_labels[:, 1]  # target task label
            preds = torch.argmax(logits, dim=-1)
            total_right += torch.sum(preds == y)
            total_samples += len(y)
        acc = float(total_right) / total_samples
    torch.cuda.empty_cache()
    return acc


def heavy_finetune_test(model_path=None):
    if model_path is None:
        model = select_model(config=args)
        num_features = model.decoder.in_features
        model.decoder = nn.Linear(num_features, 2)
    else:
        model = torch.load(model_path)
        num_features = model.decoder.in_features
        model.decoder = nn.Linear(num_features, 2)
    model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=lr,
                          weight_decay=wd,
                          momentum=momentum)
    best_test_acc = 0.0
    penalty = 0
    for _ in tqdm(range(finetune_epochs)):
        if penalty > tolerance:
            break
        model.train()
        for step, (x, multi_labels) in enumerate(auxiliary_train_loader):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            logits, embeddings = model(x)
            target_labels = multi_labels[:, 1]  # target task label
            loss = loss_fn(logits, target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_acc = test(model=model, data_loader=test_loader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            penalty = 0
        else:
            penalty += 1
        print(f"best test acc: {best_test_acc:.4f}")
    return best_test_acc * 100

def light_finetune_test(model_path=None):
    if model_path is None:
        model = select_model(config=args)
        num_features = model.decoder.in_features
        model.decoder = nn.Linear(num_features, 2)
    else:
        model = torch.load(model_path)
        num_features = model.decoder.in_features
        model.decoder = nn.Linear(num_features, 2)
    model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=lr,
                          weight_decay=wd,
                          momentum=momentum)
    for _ in tqdm(range(finetune_epochs)):
        model.train()
        for step, (x, multi_labels) in enumerate(auxiliary_train_loader):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            z, (z_mu, z_sigma) = model.featurize(x)
            logits = model.decoder(z_mu)
            target_labels = multi_labels[:, 1]  # target task label
            loss = loss_fn(logits, target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_acc = test(model=model, data_loader=test_loader)
    print(f"test acc: {test_acc:.4f}")
    return test_acc * 100

def transfer_test(model_path=None):
    print(model_path)
    results = []
    for seed in range(num_seeds):
        setup_seed(seed)
        if light_finetune:
            test_acc = light_finetune_test(model_path=model_path)
            results.append(test_acc)
            print('seed', seed, f'test acc: {test_acc}')
        else:
            best_test_acc = heavy_finetune_test(model_path=model_path)
            print('seed', seed, f'best test acc: {best_test_acc}')
            results.append(best_test_acc)
    return results

if args.use_wandb:
    import wandb

    wandb.init(project="VL", entity="tdye24")
    wandb.watch_called = False
    config = wandb.config
    config.update(args)
else:
    config = args

train_loaders, test_loader, auxiliary_train_loader = setup_datasets(config=config)
res = transfer_test(model_path=path)
print("results", res)
print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))
if args.use_wandb:
    wandb.log({
        'mean': np.mean(res),
        'std': np.std(res, ddof=1),
    })
# if args.path is None:
#     print("ImageNet Pretrained Model.")
#     res = transfer_test(model_path=None)
#     print("results", res)
#     print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))
#     if args.use_wandb:
#         wandb.log({
#             'mean': np.mean(res),
#             'std': np.std(res, ddof=1),
#         })
# else:
#     path = os.path.join('./logs', args.path)
#     if os.path.isdir(path):
#         model_names = [item for item in os.listdir(path) if 'model' in item]
#         rounds = sorted([int(item.split('-')[0]) for item in model_names], reverse=True)
#         for r in rounds:
#             model_path = os.path.join(path, f"{r}-model.pt")
#             res = transfer_test(model_path=model_path)
#             print("results", res)
#             print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))
#             if args.use_wandb:
#                 wandb.log({
#                     'mean': np.mean(res),
#                     'std': np.std(res, ddof=1),
#                 })
#
#     elif os.path.isfile(path):
#         res = transfer_test(model_path=path)
#         print(res)
#         print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))


