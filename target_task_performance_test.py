import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from utils.setup_md import setup_datasets
from torchvision import models

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
                        default=10,
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
                        default=35)

    parser.add_argument('--target-PID',
                        type=int,
                        default=15)

    parser.add_argument('--num-users',
                        type=int,
                        default=40)

    parser.add_argument('--batch-size',
                        type=int,
                        default=16)

    parser.add_argument('--mt',
                        action='store_true',
                        default=False)

    return parser.parse_args()

args = parse_args()
lr = args.lr
wd = args.wd
momentum = args.momentum
finetune_epochs = args.finetune_epochs
mt = args.mt
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


def finetune_test(model_path=None):
    if model_path is None:
        model = models.resnet18(pretrained=True)
    else:
        model = torch.load(model_path)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    if mt:
        target_head = nn.Linear(model.fc.in_features, 2)
        target_head.cuda()
        target_head.train()
        optimizer = optim.SGD(params=
                              [{'params': model.parameters()},
                               {'params': target_head.parameters()}],
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
        for step, (x, multi_labels) in enumerate(train_loaders[0]):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            logits = model(x)
            target_labels = multi_labels[:, 1]  # target task label
            loss = loss_fn(logits, target_labels)
            if mt:
                main_labels = multi_labels[:, 0]
                body = torch.nn.Sequential(*list(model.children())[:-1])
                embedding = body(x).view(x.shape[0], -1)
                main_logits = target_head(embedding)
                loss = 0.5 * loss + 0.5 * loss_fn(main_logits, main_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    acc = test(model=model, data_loader=test_loader)
    return acc * 100

def transfer_test(model_path):
    print(model_path)
    results = []
    for seed in range(5):
        setup_seed(seed)
        test_acc = finetune_test(model_path=model_path)
        print('seed', seed, 'test acc', test_acc)
        results.append(test_acc)
    return results

if args.use_wandb:
    import wandb

    wandb.init(project="VL", entity="tdye24")
    wandb.watch_called = False
    config = wandb.config
    config.update(args)
else:
    config = args

train_loaders, test_loader = setup_datasets(config=config)
if args.path is None:
    print("ImageNet Pretrained Model.")
    res = transfer_test(model_path=None)
    print(res)
    print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))
else:
    path = os.path.join('./logs', args.path)
    if os.path.isdir(path):
        model_names = os.listdir(path)
        for m_name in model_names:
            model_path=os.path.join(path, m_name)
            res = transfer_test(model_path=model_path)
            print(res)
            print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))

    elif os.path.isfile(path):
        res = transfer_test(model_path=path)
        print(res)
        print("mean", "{:.2f}".format(np.mean(res)), "std", "{:.2f}".format(np.std(res, ddof=1)))


