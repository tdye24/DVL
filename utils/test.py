import numpy as np
from models.celeba.smallvgg import Model


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
        if shape[0] * shape[1] > pool_thresh:  # MLP
            component_g = np.max(component_g, 0)

        pooled_g.append(component_g.flatten())

    return np.concatenate(pooled_g)


model = Model()
pooled_g = pool_gradient(model)
print(pooled_g.shape)


def target_test(self):
    model = deepcopy(self.model)
    body = torch.nn.Sequential(*list(model.children())[:-1])
    target_head = deepcopy(self.target_head)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [{'params': body.parameters()},
         {'params': target_head.parameters()}],
        lr=self.config.target_lr,
        momentum=0.9
    )
    for _ in tqdm(range(self.config.finetune_iterations)):
        body.train()
        target_head.train()
        for step, (x, multi_labels) in enumerate(self.auxiliary_train_loader):
            if torch.cuda.is_available():
                x, multi_labels = x.cuda(), multi_labels.cuda()
            embeddings = body(x).view(x.shape[0], -1)
            logits = target_head(embeddings)
            y = multi_labels[:, 1]  # target task label
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Valid and test
        # train_acc = self.test(model=model,
        #                       target_head=target_head,
        #                       data_loader=self.auxiliary_train_loader)
        valid_acc, _ = self.test(model=model,
                                 target_head=target_head,
                                 data_loader=self.auxiliary_valid_loader)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print("Valid", best_valid_acc)
            best_valid_model = deepcopy(model)
            best_valid_target_head = deepcopy(target_head)
            penalty = 0
        else:
            penalty += 1
    test_acc, _ = self.test(model=best_valid_model,
                            target_head=best_valid_target_head,
                            data_loader=self.test_loader)
    return best_valid_acc, test_acc