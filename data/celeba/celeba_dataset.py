import json
import os
import pprint

import torch
import random
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_dir, setup_seed
from utils.constants import *
from tqdm import tqdm

# prop_ids = [20, 31, 15, 35, 39]
prop_ids = [15, 20, 31, 39]

class CelebA_DATASET(Dataset):
    def __init__(self, hdf5_path='/home/tdye/VL/data/celeba/data_with_labels.h5',
                 image_names=None,
                 img_transform=celeba_transform,
                 attr_transform=None):
        self.hdf5_path = hdf5_path
        self.image_names = image_names
        self.img_transform = img_transform
        self.attr_transform = attr_transform
        self.hdf5_dataset = None

    def open_hdf5(self):
        self.hdf5_dataset = h5py.File(self.hdf5_path, 'r')

    def __getitem__(self, item):
        if self.hdf5_dataset is None:
            self.open_hdf5()
        img_name = self.image_names[item]
        image_np = np.array(self.hdf5_dataset[img_name + '/image'])
        all_attributes = self.hdf5_dataset[img_name].attrs['attributes']
        image, selected_attributes = None, None
        if self.img_transform is not None:
            image = self.img_transform(image_np)
        if self.attr_transform is not None:
            selected_attributes = self.attr_transform(all_attributes)
        return image, selected_attributes

    def __len__(self):
        return len(self.image_names)


def construct_prop_nonprop_img_names():
    if os.path.exists('/home/tdye/VL/data/celeba/prop_nonprop_img_names.json'):
        with open('/home/tdye/VL/data/celeba/prop_nonprop_img_names.json', 'r') as f:
            prop_nonprop_img_names = json.load(f)
    else:
        all_data_path = os.path.dirname(os.path.abspath(__file__)) + '/all'
        all_clients, all_dataset = read_dir(all_data_path)
        img_names = []
        for c in all_clients:
            img_names.extend(all_dataset[c]['x'])
        print(f"Total samples:{len(img_names)}.")
        prop_nonprop_img_names = {
            i: {
                'prop': [],
                'nonprop': []
            } for i in range(40)
        }
        hdf5_path = '/home/tdye/VL/data/celeba/data_with_labels.h5'
        file = h5py.File(hdf5_path, 'r')
        for img_name in tqdm(img_names):
            attrs = file[img_name].attrs['attributes']
            for index in range(40):
                if attrs[index] == 1:
                    prop_nonprop_img_names[index]['prop'].append(img_name)
                else:
                    prop_nonprop_img_names[index]['nonprop'].append(img_name)
        with open('/home/tdye/VL/data/celeba/prop_nonprop_img_names.json', 'w') as f:
            json.dump(obj=prop_nonprop_img_names, fp=f)
    return prop_nonprop_img_names

# def construct_training_test_img_names():
#     if os.path.exists('/home/tdye/VL/data/celeba/training_img_names.json') and \
#             os.path.exists('/home/tdye/VL/data/celeba/test_img_names.json'):
#         with open('/home/tdye/VL/data/celeba/training_img_names.json', 'r') as f:
#             training_img_names = json.load(f)
#         with open('/home/tdye/VL/data/celeba/test_img_names.json', 'r') as f:
#             test_img_names = json.load(f)
#     else:
#         res = construct_prop_nonprop_img_names()
#         training_test_img_names = set()
#         for p_id in prop_ids:
#             lst = res[str(p_id)]['prop']
#             random.shuffle(lst)
#             prop = lst[:625]
#
#             lst = res[str(p_id)]['nonprop']
#             random.shuffle(lst)
#             nonprop = lst[:625]
#
#             training_test_img_names.update(prop)
#             training_test_img_names.update(nonprop)
#
#         training_test_img_names = list(training_test_img_names)
#         random.shuffle(training_test_img_names)
#         training_img_names = training_test_img_names[:4000]
#         test_img_names = training_test_img_names[4000:]
#         with open('/home/tdye/VL/data/celeba/training_img_names.json', 'w') as f:
#             json.dump(obj=training_img_names, fp=f)
#         with open('/home/tdye/VL/data/celeba/test_img_names.json', 'w') as f:
#             json.dump(obj=test_img_names, fp=f)
#     hdf5_path = '/home/tdye/VL/data/celeba/data_with_labels.h5'
#     file = h5py.File(hdf5_path, 'r')
#     print("Training statistics.")
#     counts = {
#         i: {
#             'prop': 0,
#             'nonprop': 0
#         } for i in prop_ids
#     }
#     for img_name in tqdm(training_img_names):
#         attrs = file[img_name].attrs['attributes']
#         for index in prop_ids:
#             if attrs[index] == 1:
#                 counts[index]['prop'] += 1
#             else:
#                 counts[index]['nonprop'] += 1
#     pprint.pprint(counts)
#
#     print("Test statistics.")
#     counts = {
#         i: {
#             'prop': 0,
#             'nonprop': 0
#         } for i in prop_ids
#     }
#     for img_name in tqdm(test_img_names):
#         attrs = file[img_name].attrs['attributes']
#         for index in prop_ids:
#             if attrs[index] == 1:
#                 counts[index]['prop'] += 1
#             else:
#                 counts[index]['nonprop'] += 1
#     pprint.pprint(counts)
#     return training_img_names, test_img_names

def construct_training_test_img_names():
    if os.path.exists('/home/tdye/VL/data/celeba/training_img_names.json') and \
            os.path.exists('/home/tdye/VL/data/celeba/test_img_names.json'):
        with open('/home/tdye/VL/data/celeba/training_img_names.json', 'r') as f:
            training_img_names = json.load(f)
        with open('/home/tdye/VL/data/celeba/test_img_names.json', 'r') as f:
            test_img_names = json.load(f)
    else:
        training_test_img_names = []
        res = construct_prop_nonprop_img_names()
        main_p_id = prop_ids[0]
        lst = res[str(main_p_id)]['prop']
        random.shuffle(lst)
        prop = lst[:2500]

        lst = res[str(main_p_id)]['nonprop']
        random.shuffle(lst)
        nonprop = lst[:2500]

        training_test_img_names.extend(prop)
        training_test_img_names.extend(nonprop)

        random.shuffle(training_test_img_names)
        training_img_names = training_test_img_names[:4000]
        test_img_names = training_test_img_names[4000:]
        with open('/home/tdye/VL/data/celeba/training_img_names.json', 'w') as f:
            json.dump(obj=training_img_names, fp=f)
        with open('/home/tdye/VL/data/celeba/test_img_names.json', 'w') as f:
            json.dump(obj=test_img_names, fp=f)
    hdf5_path = '/home/tdye/VL/data/celeba/data_with_labels.h5'
    file = h5py.File(hdf5_path, 'r')
    print("Training statistics.")
    counts = {
        i: {
            'prop': 0,
            'nonprop': 0
        } for i in prop_ids
    }
    for img_name in tqdm(training_img_names):
        attrs = file[img_name].attrs['attributes']
        for index in prop_ids:
            if attrs[index] == 1:
                counts[index]['prop'] += 1
            else:
                counts[index]['nonprop'] += 1
    pprint.pprint(counts)

    print("Test statistics.")
    counts = {
        i: {
            'prop': 0,
            'nonprop': 0
        } for i in prop_ids
    }
    for img_name in tqdm(test_img_names):
        attrs = file[img_name].attrs['attributes']
        for index in prop_ids:
            if attrs[index] == 1:
                counts[index]['prop'] += 1
            else:
                counts[index]['nonprop'] += 1
    pprint.pprint(counts)
    return training_img_names, test_img_names

def prepare_training_test_loaders(num_users=4, batch_size=16):
    # setup_seed(42)
    training_img_names, test_img_names = construct_training_test_img_names()
    train_loaders = []
    random.shuffle(training_img_names)
    num_samples_per_client = len(training_img_names) // num_users
    for user_id in range(num_users):
        c_img_names = training_img_names[user_id * num_samples_per_client: (user_id + 1) * num_samples_per_client]
        c_train_d = CelebA_DATASET(image_names=c_img_names,
                                   attr_transform=lambda x: torch.tensor(x[prop_ids[0]]).long())
        c_train_loader = DataLoader(dataset=c_train_d, batch_size=batch_size, shuffle=True, num_workers=0,
                                    pin_memory=True)
        train_loaders.append(c_train_loader)

    test_d = CelebA_DATASET(image_names=test_img_names,
                            attr_transform=lambda x: torch.tensor([x[prop_ids[0]],
                                                                   x[prop_ids[1]],
                                                                   x[prop_ids[2]],
                                                                   x[prop_ids[3]]]).long())
    test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loaders, test_loader

if __name__ == '__main__':
    train_loaders, test_loader = prepare_training_test_loaders(
        num_users=4,
        batch_size=16
    )
    # res = construct_prop_nonprop_img_names()
    # get_auxiliary_dataloader(num_samples=10000)
    # get_celeba_combined_dataloader()
    # get_property_classification_auxiliary_dataloader()
    # p_res = []
    # for target_id in range(40):
    #     p = get_pearson_correlation_between_properties(property_id_A=15, property_id_B=target_id)
    #     p_res.append(p)
    # print(p_res)
    # get_face_recognition_auxiliary_dataloader(test_size=0.8, batch_size=32)
    # construct_prop_nonprop_datasets()

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # _clients, _train_loaders, _test_loaders = get_celeba_dataloaders(batch_size=64, transform=transform)
    # for _, (data, labels) in enumerate(_train_loaders[0]):
    #     print(labels)
    #
    # print("============")
    #
    # for _, (data, labels) in enumerate(_test_loaders[0]):
    #     print(labels)
