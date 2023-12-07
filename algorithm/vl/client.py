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
        if self.user_id == config.attacker_userid:
            self.is_attacker = True
            print("User", self.user_id, "is the attacker.")
        else:
            self.is_attacker = False
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

        if self.is_attacker:
            target_head = nn.Linear(self.model.fc.in_features, 2)
            target_head.cuda()
            target_head.train()
            optimizer = optim.SGD(params=
                                  [{'params': model.parameters()},
                                   {'params': target_head.parameters()}],
                                  lr=current_lr,
                                  weight_decay=1e-4,
                                  momentum=0.9)
        else:
            optimizer = optim.SGD(params=model.parameters(),
                                  lr=current_lr,
                                  weight_decay=1e-4,
                                  momentum=0.9)

        start_point = deepcopy(model)
        running_true = 0
        running_total = 0
        running_iters = 0
        running_loss = 0
        for it in range(self.config.local_iters):
            x, multi_labels = self.get_next_batch()
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            main_labels = multi_labels[:, 0]
            main_logits = model(x)
            main_preds = torch.argmax(main_logits, dim=-1)
            running_true += torch.sum(main_preds == main_labels).item()
            running_total += len(main_labels)
            loss = self.loss_ce(main_logits, main_labels)
            running_loss += loss.item()
            running_iters += 1
            if self.is_attacker:
                target_labels = multi_labels[:, 1]
                body = torch.nn.Sequential(*list(model.children())[:-1])
                embedding = body(x).view(x.shape[0], -1)
                target_logits = target_head(embedding)
                loss = (1 - self.config.alpha) * loss + self.config.alpha * self.loss_ce(target_logits, target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pseudo_gradient = get_pseudo_gradient(old_model=deepcopy(start_point), new_model=deepcopy(model), config=self.config)
        # if self.is_attacker:
        #     放大梯度
            # print("")
        model.cpu()
        torch.cuda.empty_cache()
        return self.train_samples_num, pseudo_gradient, running_true/running_total, running_loss/running_iters

    def init_local_model(self, init_model):
        for name, p in self.model.named_parameters():
            p.data = init_model.state_dict()[name]
