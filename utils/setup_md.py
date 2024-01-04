# datasets
from data.celeba.celeba_dataset import prepare_training_test_loaders
# models
from models.celeba.model import Model
# transforms
from utils.constants import *
from utils.utils import setup_seed


def setup_datasets(config):
    setup_seed(42)
    train_loaders, test_loader = prepare_training_test_loaders(num_users=config.num_users,
                                                               batch_size=config.batch_size)
    return train_loaders, test_loader
    # train_loaders, test_loader, auxiliary_train_loader = prepare_dataloaders(main_PID=config.main_PID,
    #                                                                          target_PID=config.target_PID,
    #                                                                          num_users=config.num_users,
    #                                                                          batch_size=config.batch_size,
    #                                                                          auxiliary_train_samples=config.auxiliary_train_samples)
    # return train_loaders, test_loader, auxiliary_train_loader


def select_model(config):
    model = Model(z_dim=config.z_dim,
                  num_classes=2,
                  backbone=config.backbone,
                  probabilistic=config.probabilistic)
    return model
