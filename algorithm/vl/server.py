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
        self.auxiliary_valid_loader = None
        self.config = config
        self.clients = self.setup_clients()
        self.selected_clients = []
        self.clients_gradients = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(config=self.config)
        if torch.cuda.is_available():
            self.model.cuda()

    def setup_clients(self):
        train_loaders, test_loader = setup_datasets(config=self.config)
        users = [i for i in range(len(train_loaders))]
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   config=self.config)
            for user_id in users]
        self.test_loader = test_loader
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        if self.config.attacker_userid == -1:
            return np.random.choice(self.clients, min(self.config.clients_per_round, len(self.clients)), replace=False)
        else:
            # the attacker actively participate in each round
            benign_clients = [c for c in self.clients if c.user_id != self.config.attacker_userid]
            selected_benign_clients = np.random.choice(benign_clients, min(self.config.clients_per_round, len(self.clients))-1, replace=False)
            return np.hstack((self.clients[self.config.attacker_userid], selected_benign_clients))

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        exp_log_dir = os.path.join('./logs/',
                                   f'main_{self.config.main_PID}_target_{self.config.target_PID}_{self.config.exp_note}')

        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)

        for c in self.clients:
            print(c.user_id, "Train", c.train_samples_num)
        for i in tqdm(range(self.config.num_rounds)):
            start_time = time.time()

            # if self.config.active:
            #     self.active_PC_MTL(r=i)

            self.selected_clients = self.select_clients(round_th=i)

            training_acc = 0
            training_loss = 0
            total_samples = 0

            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                c.init_local_model(deepcopy(self.model))
                train_samples_num, c_g, c_train_acc, c_train_loss = c.train(r=i)
                self.clients_gradients.append((train_samples_num, c_g))

                training_acc += c_train_acc * train_samples_num
                training_loss += c_train_loss * train_samples_num
                total_samples += train_samples_num

            training_acc /= total_samples
            training_loss /= total_samples

            aggregated_g = fed_average(self.clients_gradients)
            self.update_global_model(agg_g=aggregated_g)
            end_time = time.time()

            print(f"training costs {end_time - start_time}(s)")

            if i == 0 or (i + 1) % self.config.eval_interval == 0:
                if self.config.save_models:
                    torch.save(self.model, os.path.join(exp_log_dir, f'{i}-model.pt'))
                test_acc, test_loss = self.test(model=self.model, data_loader=self.test_loader)
                summary = {
                    "round": i,
                    "TrainAcc": training_acc,
                    "TrainLoss": training_loss,
                    "TestAcc": test_acc,
                    "TestLoss": test_loss
                }
                # if self.config.attacker_userid != -1:
                #     attacker = self.clients[self.config.attacker_userid]
                    # target_test_acc, target_test_loss = self.test(model=attacker.model,
                    #                                               target_head=attacker.target_head,
                    #                                               data_loader=self.test_loader)
                    # target_test_acc, target_test_loss = self.test(model=self.model,
                    #                                               target_head=attacker.target_head,
                    #                                               data_loader=self.test_loader)
                    # summary.update({
                    #     "TargetTestAcc": target_test_acc,
                    #     "TargetTestLoss": target_test_loss
                    # })
                if self.config.use_wandb:
                    wandb.log(summary)
                else:
                    print(summary)

            self.clients_gradients = []
            del aggregated_g
            torch.cuda.empty_cache()

    def get_next_batch(self):
        if not self.iter_auxiliary_train_loader:
            self.iter_auxiliary_train_loader = iter(self.auxiliary_train_loader)
        try:
            (X, y) = next(self.iter_auxiliary_train_loader)
        except StopIteration:
            self.iter_auxiliary_train_loader = iter(self.auxiliary_train_loader)
            (X, y) = next(self.iter_auxiliary_train_loader)
        return X, y

    # def active_PC_MTL(self, r):
    #     model = self.model
    #     body = torch.nn.Sequential(*list(model.children())[:-1])
    #     target_head = self.target_head  # finetuning induces fl training to leak more information about target task
    #     # todo each time, reinitialize the target head to enhance representation learning, prevent from over-fitting
    #     attack_lr = self.config.attack_lr * (self.config.attack_lr_decay ** r)
    #     loss_fn = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(
    #         [{'params': body.parameters()},
    #          {'params': target_head.parameters()}],
    #         lr=attack_lr,
    #         weight_decay=1e-4,
    #         momentum=0.9
    #     )
    #     body.train()
    #     target_head.train()
    #     for epoch in range(self.config.attack_iterations):
    #         x, multi_labels = self.get_next_batch()
    #         y = multi_labels[:, 1] # target task label
    #         if torch.cuda.is_available():
    #             x, y = x.cuda(), y.cuda()
    #         embedding = body(x).view(x.shape[0], -1)
    #         logits = target_head(embedding)
    #         loss = loss_fn(logits, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     torch.cuda.empty_cache()

    @staticmethod
    def test(model=None, target_head=None, data_loader=None):
        if target_head is None:
            # main task test
            model.cuda()
            model.eval()
        else:
            # target task test
            body = torch.nn.Sequential(*list(model.children())[:-1])
            body.cuda()
            body.eval()
            target_head.cuda()
            target_head.eval()

        total_right = 0
        total_samples = 0
        total_loss = 0
        total_iters = 0
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for step, (x, multi_labels) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x, multi_labels = x.cuda(), multi_labels.cuda()
                if target_head is None:
                    logits = model(x)
                    y = multi_labels[:, 0] # main task label
                else:
                    features = body(x).view(x.shape[0], -1)
                    logits = target_head(features)
                    y = multi_labels[:, 1] # target task label
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
        for (_, param), (_, param_g) in zip(self.model.named_parameters(), agg_g.named_parameters()):
            param.data = param.data - param_g.data