import os
from copy import deepcopy

import numpy as np
import torch

from utils.utils import *
from utils.constants import *
from utils.setup_md import *
from algorithm.vl.client import CLIENT
from tqdm import tqdm
import time
import datetime
import wandb
import torch.nn as nn
import torch.optim as optim


class SERVER:
    def __init__(self, config):
        self.test_loader = None
        self.auxiliary_train_loader = None
        self.iter_auxiliary_train_loader = None
        self.config = config
        self.clients = self.setup_clients()
        self.selected_clients = []
        self.num_samples_models = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(config=self.config)
        self.target_head = nn.Linear(self.model.decoder.in_features, 2)
        if torch.cuda.is_available():
            self.model.cuda()
            self.target_head.cuda()

    def setup_clients(self):
        train_loaders, test_loader, auxiliary_train_loader = setup_datasets(config=self.config)
        users = [i for i in range(len(train_loaders))]
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   config=self.config)
            for user_id in users]
        self.test_loader = test_loader
        self.auxiliary_train_loader = auxiliary_train_loader
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.clients, min(self.config.clients_per_round, len(self.clients)), replace=False)

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        exp_log_dir = os.path.join('./logs/',
                                   f'main_{self.config.main_PID}_target_{self.config.target_PID}_{self.config.exp_note}')

        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)

        for c in self.clients:
            print(c.user_id, "Train", c.train_samples_num)

        # save the randomly initialized model
        torch.save(self.model, os.path.join(exp_log_dir, f'0-model.pt'))
        if self.config.active:
            torch.save(self.target_head, os.path.join(exp_log_dir, f'0-target-head.pt'))

        for i in tqdm(range(self.config.num_rounds)):
            start_time = time.time()
            target_training_acc = 0
            if self.config.active:
                print("Active stealing.")
                target_training_acc = self.active_PC_MTL(r=i)
            self.selected_clients = self.select_clients(round_th=i)

            training_acc = 0
            training_loss = 0
            total_samples = 0

            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                c.init_local_model(deepcopy(self.model))
                train_samples_num, c_model, c_train_acc, c_train_loss = c.train(r=i)
                self.num_samples_models.append((train_samples_num, c_model))

                training_acc += c_train_acc * train_samples_num
                training_loss += c_train_loss * train_samples_num
                total_samples += train_samples_num

            training_acc /= total_samples
            training_loss /= total_samples

            averaged_model = fed_average(self.num_samples_models)
            self.model.load_state_dict(averaged_model.state_dict())
            end_time = time.time()

            print(f"training costs {end_time - start_time}(s)")

            if (i + 1) % self.config.eval_interval == 0:
                test_acc, test_loss = self.test(model=self.model, data_loader=self.test_loader)
                summary = {
                    "round": i,
                    "TrainAcc": training_acc,
                    "TrainLoss": training_loss,
                    "TestAcc": test_acc,
                    "TestLoss": test_loss,
                }
                torch.save(self.model, os.path.join(exp_log_dir, f'{i+1}-model.pt'))
                if self.config.active:
                    torch.save(self.target_head, os.path.join(exp_log_dir, f'{i+1}-target-head.pt'))
                if self.config.use_wandb:
                    wandb.log(summary)
                else:
                    print(summary)

            self.num_samples_models = []
            del averaged_model
            torch.cuda.empty_cache()

    def finetune(self, model, target_head):
        lr = 0.001
        wd = 1e-4
        momentum = 0.9
        finetune_epochs = 20
        model.fc = target_head
        model.cuda()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(),
                              lr=lr,
                              weight_decay=wd,
                              momentum=momentum)
        for _ in range(finetune_epochs):
            model.train()
            for step, (x, multi_labels) in enumerate(self.auxiliary_train_loader):
                if torch.cuda.is_available():
                    x, multi_labels = x.cuda(), multi_labels.cuda()
                logits = model(x)
                target_labels = multi_labels[:, 1]  # target task label
                loss = loss_fn(logits, target_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    def get_next_batch(self):
        if not self.iter_auxiliary_train_loader:
            self.iter_auxiliary_train_loader = iter(self.auxiliary_train_loader)
        try:
            (X, y) = next(self.iter_auxiliary_train_loader)
        except StopIteration:
            self.iter_auxiliary_train_loader = iter(self.auxiliary_train_loader)
            (X, y) = next(self.iter_auxiliary_train_loader)
        return X, y

    def active_PC_MTL(self, r):
        model = self.model
        body = model.encoder
        target_head = self.target_head  # finetuning induces fl training to leak more information about target task
        # todo each time, reinitialize the target head to enhance representation learning, prevent from over-fitting
        attack_lr = self.config.attack_lr * (self.config.attack_lr_decay ** r)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            [{'params': body.parameters()},
             {'params': target_head.parameters()}],
            lr=attack_lr,
            weight_decay=1e-4,
            momentum=0.9
        )
        body.train()
        target_head.train()
        running_true = 0
        running_total = 0
        for epoch in range(self.config.attack_iterations):
            x, multi_labels = self.get_next_batch()
            y = multi_labels[:, 1] # target task label
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            if self.config.probabilistic:
                z, (z_mu, z_sigma) = body(x)
            else:
                z = body(x)
            logits = target_head(z)
            preds = torch.argmax(z, dim=-1)
            running_true += torch.sum(preds == y).item()
            running_total += len(y)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        return running_true / running_total * 100
    @staticmethod
    def test(model=None, data_loader=None, task='main'):
        # main task test
        model.cuda()
        model.eval()

        total_right = 0
        total_samples = 0
        total_loss = 0
        total_iters = 0
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for step, (x, multi_labels) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x, multi_labels = x.cuda(), multi_labels.cuda()
                if task == 'main':
                    y = multi_labels[:, 0] # main task label
                else:
                    y = multi_labels[:, 1] # target task label
                logits = model(x)
                loss = loss_fn(logits, y)
                total_loss += loss.item()
                total_iters += 1
                preds = torch.argmax(logits, dim=-1)
                total_right += torch.sum(preds == y)
                total_samples += len(y)

            acc = float(total_right) / total_samples
            average_loss = total_loss / total_iters
        # del model
        torch.cuda.empty_cache()
        return acc, average_loss

    def update_global_model(self, agg_g):
        # for (_, param), (_, param_g) in zip(self.model.named_parameters(), agg_g.named_parameters()):
        #     param.data = param.data - param_g.data
        for (_, param), (_, param_g) in zip(self.model.state_dict().items(), agg_g.state_dict().items()):
            param.data = param.data - param_g.data