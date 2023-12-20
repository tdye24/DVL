# datasets
from data.celeba.celeba_dataset import prepare_dataloaders
# models
from models.celeba.leaf import Leaf
from models.celeba.resent18 import get_resnet18
# transforms
from utils.constants import *
from utils.utils import setup_seed


def setup_datasets(config):
    setup_seed(42)
    train_loaders, test_loader, auxiliary_train_loader = None, None, None
    # main task
    if config.main_task == 'PC':
        print(f"Main task: property classification, property: {PID_2_NAME[config.main_PID]}({config.main_PID+1})")
        train_loaders, test_loader, auxiliary_train_loader = prepare_dataloaders(main_PID=config.main_PID,
                                                                                 target_PID=config.target_PID,
                                                                                 num_users=config.num_users,
                                                                                 batch_size=config.batch_size,
                                                                                 auxiliary_train_samples=config.auxiliary_train_samples)
    else:
        assert config.main_task == 'IC'
        # print(f"Main task: identity classification, num identity: {config.main_num_identity}")
        # users, train_loaders, test_loaders = get_identity_classification_dataloaders(batch_size=config.batch_size)



    # elif dataset == 'cifar10':
    #     users, train_loaders, test_loaders = get_cifar10_data_loaders(batch_size=batch_size,
    #                                                                 transform=celeba_transform)
    # target task
    return train_loaders, test_loader, auxiliary_train_loader


def select_model(config):
    model_name = config.model
    model = None
    if model_name == 'leaf':
        if config.main_task == 'PC':
            model = Leaf(num_classes=2, z_dim=config.z_dim, probabilistic=config.probabilistic)
        else:
            assert config.main_task == 'IC'
            model = Leaf(num_classes=config.main_num_entity)

    elif model_name == 'resnet18':
        if config.main_task == 'PC':
            model = get_resnet18(pretrained=not config.no_pretrained, num_classes=2)
        else:
            assert config.main_task == 'IC'
            model = get_resnet18(pretrained=not config.no_pretrained, num_classes=config.main_num_entity)
    else:
        assert "Unimplemented model!"
    return model
