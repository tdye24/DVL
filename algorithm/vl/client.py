from utils.setup_md import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from utils.utils import adjust_learning_rate_classifier, get_pseudo_gradient


class CLIENT:
    def __init__(self, user_id, train_loader, config):
        self.config = config
        self.user_id = user_id
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")
        self.model = select_model(config=config)
        self.train_loader = train_loader
        self.iter_train_loader = None
        self.loss_ce = nn.CrossEntropyLoss()

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    def get_next_batch(self):
        if not self.iter_train_loader:
            self.iter_train_loader = iter(self.train_loader)
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X, y

    def train(self, r):
        model = self.model
        model.cuda()
        model.train()
        current_lr = self.config.main_lr * (self.config.main_lr_decay ** r)
        optimizer = optim.SGD(params=model.parameters(),
                              lr=current_lr,
                              weight_decay=1e-4,
                              momentum=0.9)
        running_true = 0
        running_total = 0
        running_iters = 0
        running_loss = 0
        for it in range(self.config.local_iters):
            x, y = self.get_next_batch()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            running_true += torch.sum(preds == y).item()
            running_total += len(y)
            loss = self.loss_ce(logits, y)
            running_loss += loss.item()
            running_iters += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.cpu()
        torch.cuda.empty_cache()
        return self.train_samples_num, deepcopy(self.model), running_true/running_total, running_loss/running_iters

    def init_local_model(self, init_model):
        for key in init_model.state_dict().keys():
            if 'num_batches_tracked' not in key:
                self.model.state_dict()[key].data.copy_(init_model.state_dict()[key])
