import os
from copy import deepcopy

import numpy as np
import torch

from utils.utils import *
from utils.constants import *
from utils.setup_md import *
from algorithm.fedavg.client import CLIENT
from tqdm import tqdm
import time
import datetime
import wandb
import torch.nn as nn
import torch.optim as optim
from data.celeba.celeba_dataset_bak import construct_prop_nonprop_datasets, CelebA_DATASET
from torch.utils.data import DataLoader


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.selected_clients = []
        self.clients_gradients = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(config=config)
        if torch.cuda.is_available():
            self.model.cuda()
        self.prop_nonprop_datasets = construct_prop_nonprop_datasets()
        self.best_validation_acc = -1
        self.best_test_acc = -1

    def setup_clients(self):
        users, train_loaders, test_loaders = setup_datasets(dataset=self.config.dataset,
                                                            batch_size=self.config.batch_size)
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   test_loader=test_loaders[user_id],
                   config=self.config)
            for user_id in users]
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.clients, self.config.clients_per_round, replace=False)

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")

        exp_dir_name = get_exp_dir_name()
        exp_log_dir = os.path.join('./grads/', exp_dir_name)
        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)
        # for c in self.clients:
        #     print("Train", c.user_id, c.train_dataset_stats)
        #     print("Test", c.user_id, c.test_dataset_stats)
        for i in tqdm(range(self.config.num_rounds)):
            # infer the selected clients' property based on clients' gradients per round
            train_pg = [[] for _ in range(NUM_PRIVATE_PROPS)]  # [[x, ..., x], ..., []] 6 x num x size(g)
            train_npg = [[] for _ in range(NUM_PRIVATE_PROPS)]
            # test_pg = [[] for _ in range(NUM_PRIVATE_PROPS)]  # [[x, ..., x], ..., []] 6 x num x size(g)
            # test_npg = [[] for _ in range(NUM_PRIVATE_PROPS)]  # [[x, ..., x], ..., []] 6 x num x size(g)
            test_g = []  # [x, x, ..., x] num_clients x size(g)

            start_time = time.time()
            round_dir = os.path.join(exp_log_dir, str(i))
            if not os.path.exists(round_dir):
                os.mkdir(round_dir)
            if i >= self.config.warm_up_rounds:
                self.construct_prop_nonprop_samples(global_model=deepcopy(self.model), train_pg=train_pg, train_npg=train_npg)

            self.selected_clients = self.select_clients(round_th=i)
            client_ids = []
            contains_prop = []
            # construct test_pg and test_npg
            for k in range(len(self.selected_clients)):
                c = self.selected_clients[k]
                print(f"{c.user_id} is training...")
                c.init_local_model(deepcopy(self.model))
                train_samples_num, c_g, loss, c_contains_prop = c.train(round_th=i)
                self.clients_gradients.append((train_samples_num, c_g))
                if i >= self.config.warm_up_rounds:
                    # for property_id in range(NUM_PRIVATE_PROPS):
                    #     pooled_c_g = self.pool_gradient(deepcopy(c_g))
                    #     if c_contains_prop[property_id]:  # the data with prop is used in training
                    #         test_pg[property_id].append(pooled_c_g)
                    #     else:
                    #         test_npg[property_id].append(pooled_c_g)
                    client_ids.append(c.user_id)
                    contains_prop.append(c_contains_prop)
                    pooled_c_g = self.pool_gradient(deepcopy(c_g))
                    test_g.append(pooled_c_g)

            aggregated_g = fed_average(self.clients_gradients)
            self.update_global_model(agg_g=aggregated_g)

            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if i == 0 or (i + 1) % self.config.eval_interval == 0:
                train_acc_list, train_loss_list, test_acc_list, test_loss_list = self.test()
                # print and log
                self.print_and_log(i, train_acc_list, train_loss_list,
                                   test_acc_list, test_loss_list)
            for item in self.clients_gradients:
                del item
            self.clients_gradients = []

            if i >= self.config.warm_up_rounds:
                np.savez(os.path.join(round_dir, "{}.npz".format("train_test")), train_pg=train_pg, train_npg=train_npg,
                         test_g=test_g, client_ids=client_ids, contains_prop=contains_prop)
            del aggregated_g
            del train_pg, train_npg, test_g
            torch.cuda.empty_cache()

    @staticmethod
    def pool_gradient(gradient, pool_thresh=5000):
        pooled_g = []
        params_names = [n for n, _ in gradient.named_parameters()]
        for name in params_names:
            component_g = np.asarray(gradient.state_dict()[name].cpu())
            shape = component_g.shape

            if len(shape) == 1:
                continue  # todo skip bias, but if the model contains BN layers?

            # component_g = np.abs(component_g)  # todo i think it is not necessary...
            if len(shape) == 4:  # CNN
                if shape[0] * shape[1] > pool_thresh:
                    continue
                component_g = component_g.reshape(shape[0], shape[1], -1)

            # if len(shape) > 2 or shape[0] * shape[1] > pool_thresh:
            #     component_g = np.max(component_g, -1)

            if len(shape) > 2:
                component_g = np.max(component_g, -1)
            if shape[0] * shape[1] > pool_thresh:
                component_g = np.max(component_g, 0)

            pooled_g.append(component_g.flatten())

        return np.concatenate(pooled_g)

    def construct_prop_nonprop_samples(self, global_model, train_pg, train_npg):
        print("Constructing prop and nonprop samples...")
        # construct prop and nonprop data on the server
        for property_id in range(NUM_PRIVATE_PROPS):
            prop_img_names = np.asarray(self.prop_nonprop_datasets[str(property_id)]['prop']['x'])
            prop_img_y = np.asarray(self.prop_nonprop_datasets[str(property_id)]['prop']['y'])
            nonprop_img_names = np.asarray(self.prop_nonprop_datasets[str(property_id)]['nonprop']['x'])
            nonprop_img_y = np.asarray(self.prop_nonprop_datasets[str(property_id)]['nonprop']['y'])
            for mix_ratio in MIX_RATIO:
                iters = 1  # default
                if mix_ratio in [1.0, 0.8, 0.6, 0.4]:
                    iters = POSITIVE_ITERS
                elif mix_ratio in [0]:
                    iters = NEGATIVE_ITERS
                for it in range(iters):  # 1 iter generates 1 sample
                    print(f"Property {property_id}, mix ratio {mix_ratio}, iters {it}/{iters}")
                    # each property, starts from the same global model
                    model = deepcopy(global_model)
                    if torch.cuda.is_available():
                        if torch.cuda.device_count() > 1:
                            model = torch.nn.DataParallel(model)
                        model.cuda()
                    model.train()
                    steps = self.config.local_iters  # simulate local training
                    start_point = deepcopy(model)
                    optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr)
                    loss_ce = nn.CrossEntropyLoss()
                    batch_size = self.config.batch_size
                    prop_size = int(batch_size * mix_ratio)
                    nonprop_size = batch_size - prop_size
                    for s in range(steps):
                        prop_indices = np.random.choice(range(len(prop_img_y)), prop_size, replace=False)
                        nonprop_indices = np.random.choice(range(len(nonprop_img_y)), nonprop_size, replace=False)
                        sampled_prop_img_names = prop_img_names[prop_indices]
                        sampled_prop_img_y = prop_img_y[prop_indices]
                        sampled_nonprop_img_names = nonprop_img_names[nonprop_indices]
                        sampled_nonprop_img_y = nonprop_img_y[nonprop_indices]

                        batch_img_names = np.hstack((sampled_prop_img_names, sampled_nonprop_img_names))
                        batch_img_y = np.vstack((sampled_prop_img_y, sampled_nonprop_img_y))

                        # load the images and labels
                        dataset = CelebA_DATASET(img_names=batch_img_names, labels=batch_img_y,
                                                 transform=celeba_transform)  # only has one batch
                        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

                        x, y = next(iter(dataloader))
                        if torch.cuda.is_available():
                            x, y = x.cuda(), y.cuda()
                        labels = y[:, -1]
                        output = model(x)
                        loss = loss_ce(output, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    pseudo_gradient = get_pseudo_gradient(deepcopy(start_point.module), deepcopy(model.module))  #
                    # old - new
                    pooled_pseudo_gradient = self.pool_gradient(pseudo_gradient)
                    if mix_ratio > 0:
                        train_pg[property_id].append(pooled_pseudo_gradient)
                    else:
                        train_npg[property_id].append(pooled_pseudo_gradient)
                    del model
                    del dataloader
                    del pseudo_gradient
                    torch.cuda.empty_cache()

    def test(self):
        train_acc_list, train_loss_list = [], []
        test_acc_list, test_loss_list = [], []
        for c in self.clients:
            print(f"{c.user_id} is testing...")
            c.init_local_model(deepcopy(self.model))
            c.performance_test()
            train_acc_list.append((c.stats['train-samples'], c.stats['train-accuracy']))
            train_loss_list.append((c.stats['train-samples'], c.stats['train-loss']))
            test_acc_list.append((c.stats['test-samples'], c.stats['test-accuracy']))
            test_loss_list.append((c.stats['test-samples'], c.stats['test-loss']))

        return train_acc_list, train_loss_list, test_acc_list, test_loss_list

    def print_and_log(self, round_th,
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

        if self.config.use_wandb:
            wandb.log(summary)
        else:
            print(summary)

    def update_global_model(self, agg_g):
        for (_, param), (_, param_g) in zip(self.model.named_parameters(), agg_g.named_parameters()):
            param.data = param.data - param_g.data
