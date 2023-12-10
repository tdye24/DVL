import torch
import numpy as np
import random
import json
import os
from collections import defaultdict
from copy import deepcopy
import datetime
import argparse

ALGORITHMS = ['fedavg', 'vl']
DATASETS = ['celeba', 'cifar10']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-note',
                        type=str,
                        default=get_exp_dir_name(),
                        help='experiment note')

    parser.add_argument('--use-wandb',
                        action='store_true',
                        default=False,
                        help='Use WandB?')

    parser.add_argument('--cuda-no',
                        help='cuda id, -1 for cpu.',
                        type=int,
                        default=0)

    parser.add_argument('--algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        default='vl')

    parser.add_argument('--num-users',
                        help='number of users',
                        type=int,
                        default=4)

    parser.add_argument('--main-task',
                        help='main task',
                        choices=['PC', 'IC'], # Property Classification/Identity Classification
                        default='PC')

    parser.add_argument('--main-PID',
                        help='property id of the main task',
                        type=int,
                        choices=list(range(40)),
                        default=0)

    parser.add_argument('--main-num-identity',
                        help='number of identity for IC main task',
                        type=int,
                        default=500)

    parser.add_argument('--target-task',
                        help='target task',
                        choices=['PC', 'IC'],  # Property Classification/Identity Classification
                        default='PC')

    parser.add_argument('--target-PID',
                        help='property id of the target task',
                        type=int,
                        choices=list(range(40)),
                        default=0)

    parser.add_argument('--auxiliary-train-samples',
                        help='num of train samples for the target task',
                        type=int,
                        default=100)

    parser.add_argument('--target-num-identity',
                        help='number of identity for IC target task',
                        type=int,
                        default=100)

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        choices=['resnet18'],
                        default='resnet18')

    parser.add_argument('--num-rounds',
                        help='# of communication round',
                        type=int,
                        default=100)

    parser.add_argument('--main-lr',
                        help='clients learning rate',
                        type=float,
                        default=0.001)

    parser.add_argument('--main-lr-decay',
                        type=float,
                        default=0.99)

    parser.add_argument('--attack-lr',
                        help='learning rate of the server during activate stealing',
                        type=float,
                        default=0.001)

    parser.add_argument('--attack-lr-decay',
                        type=float,
                        default=0.99)

    parser.add_argument('--attack-iterations',
                        help='attack iterations, too big degrades the main task, too small has no effect',
                        type=int,
                        default=10)

    parser.add_argument('--eval-interval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clients-per-round',
                        help='# of selected clients per round',
                        type=int,
                        default=4)

    parser.add_argument('--local-iters',
                        help='# of iters',
                        type=int,
                        default=1)  # sgd

    parser.add_argument('--batch-size',
                        help='batch size when clients train on data',
                        type=int,
                        default=16)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=42)

    parser.add_argument('--alpha',
                        help='multi-task: the coefficient before the target task',
                        type=float,
                        default=0.5)

    parser.add_argument('--active',
                        help='active stealing',
                        type=int,
                        default=1)

    parser.add_argument('--no-pretrained',
                        help='use the un-pretrained resnet18 as the initial model',
                        action='store_true',
                        default=False)

    parser.add_argument('--save-models',
                        help='save models',
                        action='store_true',
                        default=False)

    parser.add_argument('--noise-sigma',
                        help='dp noise',
                        type=float,
                        default=0.0)

    parser.add_argument('--attacker-userid',
                        help='the user id of the attacker',
                        type=int,
                        default=0) # -1 representing no attacker

    return parser.parse_args()


def setup_seed(rs):
    """
    set random seed for reproducing experiments
    :param rs: random seed
    :return: None
    """
    torch.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    torch.cuda.manual_seed(rs)
    np.random.seed(rs)
    random.seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    # clients = list(sorted(data.keys()))
    return clients, data


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_data = read_dir(train_data_dir)
    test_clients, test_data = read_dir(test_data_dir)
    assert train_clients.sort() == test_clients.sort()

    return train_clients, train_data, test_data


def fed_average(num_samples_models):
    client_samples = [item[0] for item in num_samples_models]
    client_models = [item[1] for item in num_samples_models]
    total_samples = sum(client_samples)
    client_weights = [item / total_samples for item in client_samples]
    averaged_model = deepcopy(client_models[0])

    for key in averaged_model.state_dict().keys():
        # num_batches_tracked is a non-trainable LongTensor and
        # num_batches_tracked are the same for all clients for the given datasets
        if 'num_batches_tracked' in key:
            averaged_model.state_dict()[key].data.copy_(client_models[0].state_dict()[key])
        else:
            tmp = torch.zeros_like(averaged_model.state_dict()[key])
            for client_idx in range(len(client_weights)):
                tmp += client_weights[client_idx] * client_models[client_idx].state_dict()[key]
            averaged_model.state_dict()[key].data.copy_(tmp)
    return averaged_model


def avg_metric(metric_list):
    total_weight = 0
    total_metric = 0
    for (samples_num, metric) in metric_list:
        total_weight += samples_num
        total_metric += samples_num * metric
    average = total_metric / total_weight

    return average


def adjust_learning_rate_classifier(epoch, init_lr=0.0001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr


def get_exp_dir_name():
    # 获取当前日期和时间
    current_datetime = datetime.datetime.now()

    # 格式化日期和时间戳
    timestamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    # 创建文件名，例如：2023-11-05-15-30-00.log
    # file_name = f"{timestamp}.log"
    return timestamp


def get_pseudo_gradient(old_model, new_model, config):  # old_model and new_model: type Model not ParallelModel
    with torch.no_grad():
        pseudo_gradient = deepcopy(old_model)
        # for name, p in pseudo_gradient.named_parameters():
        #     delta_p = old_model.state_dict()[name] - new_model.state_dict()[name]
        #
        #     noise = config.noise_sigma * torch.randn(size=delta_p.shape).cuda()
        #     delta_p = delta_p + noise
        #
        #     p.data = delta_p

        for name, p in pseudo_gradient.state_dict().items():
            delta_p = old_model.state_dict()[name] - new_model.state_dict()[name]

            noise = config.noise_sigma * torch.randn(size=delta_p.shape).cuda()
            delta_p = delta_p + noise

            p.data = delta_p
        return pseudo_gradient

def convert_to_image(obj, mean=0.5, std=0.5): # single image
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Please install Pillow to use images: pip install Pillow")

    if not isinstance(obj, np.ndarray):
        obj = obj.cpu().detach().numpy()
    obj = obj.transpose((1, 2, 0))
    obj = obj * std + mean
    obj = np.clip(obj * 256, 0, 255)
    obj = Image.fromarray(np.uint8(obj))
    obj.save('./converted_image.png')
    return obj