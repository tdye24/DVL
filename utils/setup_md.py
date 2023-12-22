# datasets
from data.celeba.celeba_dataset import prepare_training_test_loaders
# models
from models.celeba.leaf import Leaf
from models.celeba.resent18 import get_resnet18
# transforms
from utils.constants import *
from utils.utils import setup_seed


def setup_datasets(config):
    setup_seed(config.seed)
    # main task
    print(f"Main task: property classification, property: {PID_2_NAME[config.main_PID]}({config.main_PID+1})")
    train_loaders, test_loader = prepare_training_test_loaders(main_PID=config.main_PID,
                                                               num_users=config.num_users,
                                                               batch_size=config.batch_size)
        # print(f"Main task: identity classification, num identity: {config.main_num_identity}")
        # users, train_loaders, test_loaders = get_identity_classification_dataloaders(batch_size=config.batch_size)



    # elif dataset == 'cifar10':
    #     users, train_loaders, test_loaders = get_cifar10_data_loaders(batch_size=batch_size,
    #                                                                 transform=celeba_transform)
    # target task
    return train_loaders, test_loader


def select_model(config):
    model_name = config.model
    model = None
    if model_name == 'leaf':
        model = Leaf(num_classes=2, z_dim=config.z_dim, probabilistic=config.probabilistic)

    elif model_name == 'resnet18':
        model = get_resnet18(pretrained=not config.no_pretrained, num_classes=2)
    else:
        assert "Unimplemented model!"
    return model
