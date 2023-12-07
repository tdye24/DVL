import os.path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# from distributed_sgd import SAVE_DIR
import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import read_dir, setup_seed
from torchvision.transforms import transforms
from constants import NUM_PRIVATE_PROPS
from vis import vis


def inference_attack(data, norm=True, scale=True, figname=""):
    train_pg, train_npg, test_pg, test_npg = data

    train_pg = np.asarray(train_pg)
    train_npg = np.asarray(train_npg)
    test_pg = np.asarray(test_pg)
    test_npg = np.asarray(test_npg)
    print("train ps-nps {}-{} ** test ps-nps {}-{}".format(train_pg.shape, train_npg.shape, test_pg.shape,
                                                           test_npg.shape))

    X_train = np.vstack([train_pg, train_npg])
    y_train = np.concatenate([np.ones(len(train_pg)), np.zeros(len(train_npg))])

    X_test = np.vstack([test_pg, test_npg])
    y_test = np.concatenate([np.ones(len(test_pg)), np.zeros(len(test_npg))])

    # X_train = np.abs(X_train)  # todo need to np.abs?
    # X_test = np.abs(X_test)  # todo need to np.abs?

    if norm:
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    # store the fig
    # vis(X_train, y_train, f'{figname}.png')

    clf = RandomForestClassifier(n_estimators=200, n_jobs=5, min_samples_leaf=5, min_samples_split=5)

    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print('AUC: ', roc_auc_score(y_true=y_test, y_score=y_score))


def evaluate(exp_log_dir=''):
    with np.load(os.path.join(exp_log_dir, 'train_test.npz'), allow_pickle=True) as f:
        train_pg, train_npg, test_g, client_ids, contains_prop = f['train_pg'], f['train_npg'], f['test_g'], f[
            'client_ids'], f['contains_prop']
    # client ids is for user-level property inference, instead of round-level or communication-level
    for prop_id in range(NUM_PRIVATE_PROPS):
        try:
            print("Property: ", prop_id)
            prop_indices = contains_prop[:, prop_id] == 1
            nonprop_indices = contains_prop[:, prop_id] == 0
            sub_train_pg, sub_train_npg = train_pg[prop_id], train_npg[prop_id]
            sub_test_pg = test_g[prop_indices]
            sub_test_npg = test_g[nonprop_indices]
            inference_attack(data=(sub_train_pg, sub_train_npg, sub_test_pg, sub_test_npg),
                             figname=f"{exp_log_dir}/{prop_id}")
        except:
            # print("Dimension Error.")
            pass


def evaluate_across_rounds(exp_log_dir=''):
    # client ids is for user-level property inference, instead of round-level or communication-level
    for prop_id in range(NUM_PRIVATE_PROPS):
        print("Property: ", prop_id)
        sub_train_pg, sub_train_npg, sub_test_pg, sub_test_npg = [], [], [], []
        for r in range(100):
            try:
                with np.load(os.path.join(exp_log_dir, f'{r}/train_test.npz'), allow_pickle=True) as f:
                    train_pg, train_npg, test_g, client_ids, contains_prop = f['train_pg'], f['train_npg'], f['test_g'], \
                        f['client_ids'], f['contains_prop']
                prop_indices = contains_prop[:, prop_id] == 1
                nonprop_indices = contains_prop[:, prop_id] == 0
                sub_train_pg.append(train_pg[prop_id])
                sub_train_npg.append(train_npg[prop_id])

                sub_test_pg.append(test_g[prop_indices])
                sub_test_npg.append(test_g[nonprop_indices])
            except:
                # print("Dimension Error.")
                pass
        inference_attack(data=(np.concatenate(sub_train_pg),
                               np.concatenate(sub_train_npg),
                               np.concatenate(sub_test_pg),
                               np.concatenate(sub_test_npg)), figname=f"{exp_log_dir}/{prop_id}")


if __name__ == '__main__':
    # evaluate(exp_log_dir='./grads/2023-11-09-11-52-13/99')
    # evaluate_across_rounds(exp_log_dir='./grads/2023-11-09-11-56-51')
    evaluate_across_rounds(exp_log_dir='./grads/2023-11-09-11-52-13')
