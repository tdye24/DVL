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
        self.config = config
        self.clients = self.setup_clients()
        self.selected_clients = []
        self.clients_gradients = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(config=self.config)
        self.target_head = nn.Linear(self.model.fc.in_features, 2)
        if torch.cuda.is_available():
            self.model.cuda()
            self.target_head.cuda()
        self.test_loader = None
        self.auxiliary_train_loader = None
        self.auxiliary_valid_loader = None
        self.iter_auxiliary_train_loader = None

        self.best_target_valid_acc = -1
        self.corresponding_target_test_acc = -1

    def setup_clients(self):
        train_loaders, test_loader, auxiliary_train_loader, auxiliary_valid_loader = setup_datasets(config=self.config)
        users = [i for i in range(len(train_loaders))]
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   config=self.config)
            for user_id in users]
        self.test_loader = test_loader
        self.auxiliary_train_loader = auxiliary_train_loader
        self.auxiliary_valid_loader = auxiliary_valid_loader
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.clients, self.config.clients_per_round, replace=False)

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        exp_log_dir = os.path.join('./logs/', f'Main[{self.config.main_PID}]_Target[{self.config.target_PID}]_{self.config.exp_note}')

        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)

        for c in self.clients:
            print(c.user_id, "Train", c.train_samples_num)
        for i in tqdm(range(self.config.num_rounds)):
            start_time = time.time()

            if self.config.active:
                self.active_PC_MTL()

            self.selected_clients = self.select_clients(round_th=i)
            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                c.init_local_model(deepcopy(self.model))
                train_samples_num, c_g = c.train(round_th=i)
                self.clients_gradients.append((train_samples_num, c_g))
            aggregated_g = fed_average(self.clients_gradients)
            self.update_global_model(agg_g=aggregated_g)
            end_time = time.time()

            print(f"training costs {end_time - start_time}(s)")

            if i == 0 or (i + 1) % self.config.eval_interval == 0:
                if self.config.save_models:
                    torch.save(self.model, os.path.join(exp_log_dir, f'seed{self.config.seed}-round{i}-model.pt'))
                    torch.save(self.target_head, os.path.join(exp_log_dir, f'seed{self.config.seed}-round{i}-target-head.pt'))
                train_acc_list, train_loss_list, test_acc_list, test_loss_list = self.test()
                summary = self.summary(i, train_acc_list, train_loss_list,
                                             test_acc_list, test_loss_list)
                # At first round (0), it degrades to performing transfer learning on ResNet18 pretrained on ImageNet
                target_valid_acc, target_test_acc = self.target_test()
                if target_valid_acc >= self.best_target_valid_acc:
                    self.best_target_valid_acc = target_valid_acc
                    self.corresponding_target_test_acc = target_test_acc

                summary.update({
                    'TargetValidAcc': target_valid_acc,
                    'TargetTestAcc': target_test_acc,
                    'BestTargetValidAcc': self.best_target_valid_acc,
                    'CorrespondingTargetTestAcc': self.corresponding_target_test_acc
                })

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

    def active_PC_MTL(self):
        model = self.model
        body = torch.nn.Sequential(*list(model.children())[:-1])
        target_head = self.target_head  # finetuning induces fl training to leak more information about target task

        attack_lr = self.config.attack_lr
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            [{'params': body.parameters()},
             {'params': target_head.parameters()}],
            lr=attack_lr,
            momentum=0.9
        )
        model.train()
        target_head.train()
        for epoch in range(self.config.attack_iterations):
            x, y = self.get_next_batch()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            embedding = body(x).view(x.shape[0], -1)
            logits = target_head(embedding)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()

    def active_IC_MTL(self):
        if self.config.independent:
            model = deepcopy(self.model)
            target_head = deepcopy(self.target_head)  # finetuning 不影响 fl training
        else:
            model = self.model
            target_head = self.target_head  # finetuning induces fl training to leak more information about target task

        model.cuda()
        target_head.cuda()

        server_lr = self.config.server_lr
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            [{'params': model.parameters()}, {'params': target_head.parameters()}],
            lr=server_lr,
            weight_decay=1e-4)
        best_valid_acc = -1
        tolerance = 20
        penalty = 0
        for epoch in range(10000):
            if penalty >= tolerance:
                break
            model.train()
            target_head.train()
            for step, (x, y) in enumerate(self.auxiliary_train_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                features = model.conv_layers(x).view(x.shape[0], -1)
                target_logits = target_head(features)
                target_loss = loss_fn(target_logits, y)
                optimizer.zero_grad()
                target_loss.backward()
                optimizer.step()

            # Valid and test
            valid_acc = self.target_test(model=model,
                                         target_head=target_head,
                                         data_loader=self.auxiliary_valid_loader)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                penalty = 0
                # print(f"Best valid acc {best_valid_acc}, test acc {test_acc}")
            else:
                penalty += 1
            torch.cuda.empty_cache()
        return model, target_head, best_valid_acc

    def test(self):
        train_acc_list, train_loss_list = [], []
        test_acc_list, test_loss_list = [], []
        for c in self.clients:
            # print(f"{c.user_id} is testing...")
            c.init_local_model(deepcopy(self.model))
            c.performance_test()
            train_acc_list.append((c.stats['train-samples'], c.stats['train-accuracy']))
            train_loss_list.append((c.stats['train-samples'], c.stats['train-loss']))
            test_acc_list.append((c.stats['test-samples'], c.stats['test-accuracy']))
            test_loss_list.append((c.stats['test-samples'], c.stats['test-loss']))

        return train_acc_list, train_loss_list, test_acc_list, test_loss_list

    def target_test(self):
        model = deepcopy(self.model)
        body = torch.nn.Sequential(*list(model.children())[:-1])
        target_head = deepcopy(self.target_head)
        # test_res = {}

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            [{'params': body.parameters()},
             {'params': target_head.parameters()}],
            lr=self.config.target_lr,
            momentum=0.9
        )
        best_valid_acc = -1
        best_valid_body = None
        best_valid_target_head = None
        tolerance = 20
        penalty = 0
        for e in tqdm(range(1000)):
            if penalty > tolerance:
                break
            body.train()
            target_head.train()
            for step, (x, y, identity) in enumerate(self.auxiliary_train_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                embeddings = body(x).view(x.shape[0], -1)
                logits = target_head(embeddings)
                if self.config.target_task == 'PC':
                    loss = loss_fn(logits, y)
                else:
                    loss = loss_fn(logits, identity)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Valid and test
            train_acc = self.target_dataloader_test(body=body,
                                                    target_head=target_head,
                                                    data_loader=self.auxiliary_train_loader)
            valid_acc = self.target_dataloader_test(body=body,
                                                    target_head=target_head,
                                                    data_loader=self.auxiliary_valid_loader)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                print("Train", train_acc, "Valid", best_valid_acc)
                best_valid_body = deepcopy(body)
                best_valid_target_head = deepcopy(target_head)
                penalty = 0
            else:
                penalty += 1
        test_acc = self.target_dataloader_test(body=best_valid_body,
                                               target_head=best_valid_target_head,
                                               data_loader=self.auxiliary_test_loader)
        return best_valid_acc, test_acc

    def target_dataloader_test(self, body=None, target_head=None, data_loader=None):
        body.eval()
        target_head.eval()

        total_right = 0
        total_samples = 0
        with torch.no_grad():
            for step, (x, y, identity) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                features = body(x).view(x.shape[0], -1)
                logits = target_head(features)
                output = torch.argmax(logits, dim=-1)
                if self.config.target_task == 'PC':
                    total_right += torch.sum(output == y)
                    total_samples += len(y)
                else:
                    total_right += torch.sum(output == identity)
                    total_samples += len(identity)
            # todo use other metric besides accuracy
            acc = float(total_right) / total_samples
        # del model
        torch.cuda.empty_cache()
        return acc

    @staticmethod
    def summary(round_th,
                      gm_train_acc_list, gm_train_loss_list,
                      gm_test_acc_list, gm_test_loss_list):
        gm_train_acc = avg_metric(gm_train_acc_list)
        gm_train_loss = avg_metric(gm_train_loss_list)
        gm_test_acc = avg_metric(gm_test_acc_list)
        gm_test_loss = avg_metric(gm_test_loss_list)

        # post data error, encoder error, trainingAcc. format
        summary = {
            "round": round_th,
            "TrainAcc": gm_train_acc,
            "TestAcc": gm_test_acc,
            "TrainLoss": gm_train_loss,
            "TestLoss": gm_test_loss,

        }

        # if self.config.use_wandb:
        #     wandb.log(summary)
        # else:
        #     print(summary)
        return summary

    def update_global_model(self, agg_g):
        for (_, param), (_, param_g) in zip(self.model.named_parameters(), agg_g.named_parameters()):
            param.data = param.data - param_g.data

    def test(self, model=None, data_loader=None):
        # if torch.cuda.is_available():
        #     if torch.cuda.device_count() > 1:
        #         model = torch.nn.DataParallel(model)
        model.cuda()
        model.eval()

        total_right = 0
        total_samples = 0
        mean_loss = []
        with torch.no_grad():
            for step, (x, y, identity) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x, y, identity = x.cuda(), y.cuda(), identity.cuda()
                output = model(x)
                if self.config.main_task == 'PC':
                    loss = self.loss_ce(output, y)
                    total_right += torch.sum(torch.argmax(output, dim=-1) == y)
                    total_samples += len(y)
                else:
                    assert self.config.main_task == 'IC'
                    loss = self.loss_ce(output, identity)
                    total_right += torch.sum(torch.argmax(output, dim=-1) == identity)
                    total_samples += len(y)
                mean_loss.append(loss.item())
            acc = float(total_right) / total_samples
        model.cpu()
        torch.cuda.empty_cache()
        return total_samples, acc, sum(mean_loss) / len(mean_loss)