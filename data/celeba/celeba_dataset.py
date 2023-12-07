import json
import os
import torch
import random
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_dir, setup_seed
from utils.constants import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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
        with open('./all/prop_nonprop_img_names.json', 'w') as f:
            json.dump(obj=prop_nonprop_img_names, fp=f)
    return prop_nonprop_img_names

def prepare_dataloaders(main_PID, target_PID, num_users=40, batch_size=16):
    res = construct_prop_nonprop_img_names()
    main_prop = res[str(main_PID)]['prop']
    main_nonprop = res[str(main_PID)]['nonprop']

    target_prop = res[str(target_PID)]['prop']
    target_nonprop = res[str(target_PID)]['nonprop']

    PP = list(set(main_prop) & set(target_prop))
    PN = list(set(main_prop) & set(target_nonprop))

    NP = list(set(main_nonprop) & set(target_prop))
    NN = list(set(main_nonprop) & set(target_nonprop))

    min_samples = min(len(PP), len(PN), len(NP), len(NN))
    print("Min sample", min_samples)
    PP = np.random.choice(PP, min_samples, replace=False)
    PN = np.random.choice(PN, min_samples, replace=False)
    NP = np.random.choice(NP, min_samples, replace=False)
    NN = np.random.choice(NN, min_samples, replace=False)

    # img_name_to_labels dic, main task labels
    img_name_2_label_dic = {}
    for img_name in PP:
        img_name_2_label_dic[img_name] = 1
    for img_name in PN:
        img_name_2_label_dic[img_name] = 1
    for img_name in NP:
        img_name_2_label_dic[img_name] = 0
    for img_name in NN:
        img_name_2_label_dic[img_name] = 0

    combined_img_names = np.hstack((PP, PN, NP, NN))
    assert len(combined_img_names) >= 5000 # 4000 train + 1000 test, 4 clients and 1000 per client
    train_test_img_names = np.random.choice(combined_img_names, 5000)
    train_test_labels = [img_name_2_label_dic[img_name] for img_name in train_test_img_names]
    train_img_names, test_img_names, _, _ = train_test_split(train_test_img_names,
                                                             train_test_labels,
                                                             test_size=1000/5000,
                                                             random_state=42,
                                                             stratify=train_test_labels)

    train_loaders = []
    random.shuffle(train_img_names)
    num_samples_per_client = len(train_img_names) // num_users
    for user_id in range(num_users):
        c_img_names = train_img_names[user_id * num_samples_per_client: (user_id+1) * num_samples_per_client]
        c_train_d = CelebA_DATASET(image_names=c_img_names,
                                 attr_transform=lambda x: torch.tensor([x[main_PID], x[target_PID]]).long())
        c_train_loader = DataLoader(dataset=c_train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        train_loaders.append(c_train_loader)

    test_d = CelebA_DATASET(image_names=test_img_names,
                            attr_transform=lambda x: torch.tensor([x[main_PID], x[target_PID]]).long())
    test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loaders, test_loader


# if __name__ == '__main__':
    # train_loaders, test_loader, auxiliary_train_loader, auxiliary_valid_loader = img_names = prepare_dataloaders(main_PID=35, target_PID=15)
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
