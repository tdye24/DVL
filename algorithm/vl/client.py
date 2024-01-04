from utils.setup_md import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy


class CLIENT:
    def __init__(self, user_id, train_loader, config):
        self.config = config
        self.user_id = user_id
        self.device = torch.device(f"cuda:{config.cuda_no}") if config.cuda_no != -1 else torch.device("cpu")
        self.model = select_model(config=config)
        self.train_loader = train_loader
        self.iter_train_loader = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.probabilistic = config.probabilistic
        # self.reference_mu = nn.Parameter(torch.zeros(config.num_classes, config.z_dim).cuda())
        # self.reference_sigma = nn.Parameter(torch.ones(config.num_classes, config.z_dim).cuda())
        self.reference_mu = nn.Parameter(torch.zeros(config.z_dim).cuda())
        self.reference_sigma = nn.Parameter(torch.ones(config.z_dim).cuda())
        self.C = nn.Parameter(torch.ones([]).cuda())

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
        current_lr = self.config.lr * (self.config.lr_decay ** r)
        current_beta = self.config.beta
        optimizer = optim.SGD(params=model.parameters(),
                              lr=current_lr,
                              weight_decay=1e-4,
                              momentum=0.9)
        # if self.probabilistic:
        #     optimizer.add_param_group(
        #         {'params': [self.reference_mu, self.reference_sigma, self.C],
        #          'lr': current_lr,
        #          'momentum': 0.9})

        running_true = 0
        running_total = 0
        running_iters = 0
        running_class_loss = 0
        running_CMI_Reg = 0
        for it in range(self.config.local_iters):
            x, y = self.get_next_batch()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            (z_mu, z_sigma), z, logits = model(x)

            preds = torch.argmax(logits, dim=-1)
            running_true += torch.sum(torch.eq(preds, y)).item()
            running_total += len(y)

            class_loss = self.loss_fn(logits, y)
            running_class_loss += class_loss.item()
            if self.config.probabilistic:
                # reference_sigma_softplus = F.softplus(self.reference_sigma)
                # reference_mu = self.reference_mu[y]
                # reference_sigma = reference_sigma_softplus[y]
                reference_mu = self.reference_mu
                reference_sigma = F.softplus(self.reference_sigma)
                # z_mu_scaled = z_mu * self.C
                # z_sigma_scaled = z_sigma * self.C
                z_mu_scaled = z_mu * 1
                z_sigma_scaled = z_sigma * 1
                CMI_Reg = torch.log(reference_sigma) - torch.log(z_sigma_scaled) + \
                         (z_sigma_scaled ** 2 + (z_mu_scaled - reference_mu) ** 2) / (2 * reference_sigma ** 2) - 0.5
                CMI_Reg = CMI_Reg.sum(1).mean()
                running_CMI_Reg += CMI_Reg.item()
                total_loss = class_loss + current_beta * CMI_Reg
            else:
                total_loss = class_loss

            running_iters += 1
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.cpu()
        torch.cuda.empty_cache()
        return self.train_samples_num, \
               deepcopy(self.model), \
               self.reference_mu.clone(), \
               self.reference_sigma.clone(), \
               self.C.clone(), \
               running_true/running_total, \
               running_class_loss/running_iters, \
               running_CMI_Reg/running_iters

    def init_local_model(self, init_model, reference_mu=None, reference_sigma=None, C=None):
        for key in init_model.state_dict().keys():
            if 'num_batches_tracked' not in key:
                self.model.state_dict()[key].data.copy_(init_model.state_dict()[key])

        if self.probabilistic:
            self.reference_mu.data = reference_mu.data
            self.reference_sigma.data = reference_sigma.data
            self.C.data = C.data
