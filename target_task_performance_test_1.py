import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from utils.setup_md import setup_datasets, select_model

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

    parser.add_argument('--cuda-no',
                        type=int,
                        default=1)

    parser.add_argument('--linear-probing',
                        type=int,
                        default=1)

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
    model = torch.load(model_path)
    num_features = model.decoder.in_features
    model.decoder = nn.Linear(num_features, 2)
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
            if model.probabilistic:
                z, (z_mu, z_sigma) = model.featurize(x)
            else:
                z_mu = model.featurize(x)
            logits = model.decoder(z_mu)
            target_labels = multi_labels[:, 1]  # target task label
            loss = loss_fn(logits, target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_acc = test(model=model, data_loader=test_loader)
    # print(f"test acc: {test_acc:.4f}")
    return test_acc

def transfer_test(dir_name):
    dir_path = os.path.join('./logs/', dir_name)
    names = os.listdir(dir_path)
    rounds = sorted([int(item.split('-')[0]) for item in names])
    for r in rounds:
        model_path = os.path.join(dir_path, f'{r}-model.pt')
        # print(model_path)
        results = []
        for seed in range(num_seeds):
            setup_seed(seed)
            test_acc = finetune_test(model_path=model_path)
            results.append(test_acc)
        if args.use_wandb:
            wandb.log({
                'mean': np.mean(results),
                'std': np.std(results, ddof=1),
            })
        else:
            print("mean", "{:.4f}".format(np.mean(results)),
                  "std", "{:.4f}".format(np.std(results, ddof=1)))

if args.use_wandb:
    import wandb

    wandb.init(project="VL", entity="tdye24")
    wandb.watch_called = False
    config = wandb.config
    config.update(args)
else:
    config = args

train_loaders, test_loader, auxiliary_train_loader = setup_datasets(config=config)
transfer_test(dir_name=dir_name)