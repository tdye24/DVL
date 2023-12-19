from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.utils import *
from utils.setup_md import setup_datasets, select_model
from torchvision import models
from data.celeba.celeba_dataset import CelebA_DATASET
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
                        default=2000,
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

    parser.add_argument('--tolerance',
                        type=int,
                        default=20)

    parser.add_argument('--cuda-no',
                        type=int,
                        default=1)

    parser.add_argument('--K',
                        type=int,
                        default=5)

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

train_loaders, test_loader, auxiliary_train_loader = setup_datasets(config=config)
# 获取数据集
all_train_img_names = []
for loader in train_loaders:
    all_train_img_names.extend(loader.dataset.image_names)

all_train_d = CelebA_DATASET(image_names=all_train_img_names,
                        attr_transform=lambda x: torch.tensor([x[args.main_PID], x[args.target_PID]]).long())
all_train_loader = DataLoader(dataset=all_train_d, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


setup_seed(rs=42)

for r in range(200):
    path = f'./logs/{args.path}/{r}-model.pt'
    if os.path.exists(path):
        model = torch.load(path)
    else:
        continue
    print("model.probabilistic", bool(model.probabilistic))
    model.cuda()
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for step, (inputs, multi_labels) in tqdm(enumerate(all_train_loader)):
            inputs = inputs.cuda()
            z = model.featurize(inputs, num_samples=model.num_samples, return_dist=False)
            if model.probabilistic and model.num_samples > 1:
                z = z.view([model.num_samples, -1, z.shape[-1]]).mean(0)
            batch_labels = np.array(multi_labels[:, 1])
            embeddings.append(z.cpu().detach().numpy())
            labels.append(batch_labels)


    X_train = np.concatenate(embeddings)
    y_train = np.concatenate(labels)

    # 转换为 PyTorch 的 Tensor
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).long()

    print(f"Len={len(y_train_tensor)}")

    # 定义 kNN 分类器并拟合训练数据
    knn = KNeighborsClassifier(n_neighbors=args.K)  # 这里选择了 k=3
    knn.fit(X_train, y_train)

    # 预测测试集的标签
    y_pred = knn.predict(X_train_tensor)

    # 计算准确率
    accuracy = accuracy_score(y_train_tensor, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    if args.use_wandb:
        wandb.log({
            'round': r,
            'accuracy': accuracy
        })
