from utils.setup_md import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from utils.utils import adjust_learning_rate_classifier, get_pseudo_gradient


class CLIENT:
    def __init__(self, user_id, train_loader, test_loader, config):
        self.config = config
        self.user_id = user_id
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")
        self.model = select_model(config=config)
        self.train_loader = train_loader
        self.iter_train_loader = None
        self.iter_validation_loader = None
        self.test_loader = test_loader
        self.iter_test_loader = None
        self.loss_ce = nn.CrossEntropyLoss()
        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'train-accuracy': 0,
            'test-accuracy': 0,
            'train-loss': None,
            'test-loss': None
        }

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    # @property
    # def train_dataset_stats(self):
    #     if self.train_loader:
    #         try:
    #             Y = self.train_loader.dataset.Y[self.train_loader.dataset.ids]
    #         except:
    #             Y = self.train_loader.dataset.Y
    #         unique_labels = np.unique(Y)
    #         res = {}
    #         for l in unique_labels:
    #             count = len(np.where(Y == l)[0])
    #             res.update({l: count})
    #     else:
    #         res = "Surrogate!"
    #     return res

    def get_next_batch(self):
        if not self.iter_train_loader:
            self.iter_train_loader = iter(self.train_loader)
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X, y

    def get_next_test_batch(self):
        if not self.iter_test_loader:
            self.iter_test_loader = iter(self.test_loader)
        try:
            (X, y) = next(self.iter_test_loader)
        except StopIteration:
            self.iter_test_loader = iter(self.test_loader)
            (X, y) = next(self.iter_test_loader)
        return X, y

    def train(self, round_th):
        model = self.model
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()
        model.train()

        # current_lr = adjust_learning_rate_classifier(round_th, self.config.lr)  # todo
        # optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-4)  # todo
        current_lr = self.config.lr
        optimizer = optim.SGD(params=model.parameters(), lr=current_lr)
        mean_loss = []
        start_point = deepcopy(model)
        contains_prop_over_batches = [[] for _ in range(NUM_PRIVATE_PROPS)]  # [[True, False], ..., [True, False]] 6 x num_batches
        # (2 here)
        for it in range(self.config.local_iters):
            x, y = self.get_next_batch()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # current batch has prop or not?

            logits = model(x)

            # main task labels
            labels = y[:, -1]
            if torch.cuda.is_available():
                labels = labels.cuda()

            loss = self.loss_ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            mean_loss.append(loss.item())
            for prop_id in range(NUM_PRIVATE_PROPS):
                if torch.sum(y[:, prop_id]) >= 1:  # once there exist >= 1 images contain the prop, the batch True,
                    # else False
                    contains_prop_over_batches[prop_id].append(True)
                else:
                    contains_prop_over_batches[prop_id].append(False)

        pseudo_gradient = get_pseudo_gradient(old_model=deepcopy(start_point.module), new_model=deepcopy(model.module))
        model.cpu()
        torch.cuda.empty_cache()
        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            return 0, pseudo_gradient, sum(mean_loss) / len(mean_loss)
            # exit(0)
        # with multi-steps, if there is >= 1 batches that contains the prop, the client True, else False
        contains_prop = [1.0 for _ in range(NUM_PRIVATE_PROPS)]  # [True, ..., False] 6
        for prop_id in range(NUM_PRIVATE_PROPS):
            contains_prop[prop_id] = 1.0 if True in contains_prop_over_batches[prop_id] else 0.0
        return self.train_samples_num, pseudo_gradient, sum(mean_loss) / len(mean_loss), contains_prop

    def performance_test(self):
        train_samples, train_acc, train_loss = self.test(model=self.model, data_loader=self.train_loader)
        test_samples, test_acc, test_loss = self.test(model=self.model, data_loader=self.test_loader)

        self.stats.update({
            'train-samples': train_samples,
            'train-accuracy': train_acc,
            'train-loss': train_loss,
            'test-samples': test_samples,
            'test-accuracy': test_acc,
            'test-loss': test_loss
        })

    def test(self, model=None, data_loader=None):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()
        model.eval()

        total_right = 0
        total_samples = 0
        mean_loss = []
        with torch.no_grad():
            for step, (x, y) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                output = model(x)

                # main task labels
                labels = y[:, -1]
                if torch.cuda.is_available():
                    labels = labels.cuda()

                loss = self.loss_ce(output, labels)
                mean_loss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples
        model.cpu()
        torch.cuda.empty_cache()
        return total_samples, acc, sum(mean_loss) / len(mean_loss)

    def init_local_model(self, init_model):
        for name, p in self.model.named_parameters():
            p.data = init_model.state_dict()[name]

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
