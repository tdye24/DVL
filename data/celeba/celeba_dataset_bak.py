import json
import os
import pickle
import torch
import random
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_dir, setup_seed
from utils.constants import *
from tqdm import tqdm
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
setup_seed(rs=42)
celeba_dir = os.path.dirname(os.path.abspath(__file__)) + '/clients'
train_path = os.path.join(celeba_dir, 'train')
test_path = os.path.join(celeba_dir, 'test')
train_clients, train_dataset = read_dir(train_path)
test_clients, test_dataset = read_dir(test_path)
assert train_clients.sort() == test_clients.sort()

# def load_celeba():
#     data_dir = '/home/tdye/VL/data/celeba'  # 存储分割数据的文件夹路径
#     # 创建一个空字典来存储拼接后的数据
#     full_data_dict = {}
#     # 加载每个分割文件并拼接数据
#     files = [f_n for f_n in os.listdir(data_dir) if 'data_part' in f_n]
#     print(f"Loading {len(files)} data parts of CelebA.")
#     for f_name in tqdm(files):
#         file_path = os.path.join(data_dir, f_name)
#         # 读取当前文件中的数据
#         with open(file_path, 'rb') as f:
#             part_data = pickle.load(f)
#         # 将当前文件中的数据添加到完整的数据字典中
#         full_data_dict.update(part_data)
#     print(f"Size={len(full_data_dict.keys())}")
#     return full_data_dict
#
# FULL_DATA_DIC = load_celeba()

def construct_prop_nonprop_datasets(property_id):
    all_data_path = os.path.dirname(os.path.abspath(__file__)) + '/all'
    all_clients, all_dataset = read_dir(all_data_path)
    public_dataset_img_names = []
    for c in all_clients:
        if c not in train_clients:
            public_dataset_img_names.extend(all_dataset[c]['x'])
    print(f"Total samples in public dataset :{len(public_dataset_img_names)}.")
    prop_nonprop_img_names = {
        'prop': [],
        'nonprop': []
    }
    hdf5_path = '/home/tdye/VL/data/celeba/data_with_labels.h5'
    file = h5py.File(hdf5_path, 'r')
    for img_name in public_dataset_img_names:
        target_att = file[img_name].attrs['attributes'][property_id]
        if target_att == 1:  # prop
            prop_nonprop_img_names['prop'].append(img_name)
        else:  # nonprop
            prop_nonprop_img_names['nonprop'].append(img_name)
    return prop_nonprop_img_names


class CelebA_DATASET(Dataset):
    def __init__(self, hdf5_path='/home/tdye/VL/data/celeba/data_with_labels.h5',
                 image_names=None,
                 img_transform=celeba_transform,
                 attr_transform=None,
                 specified_identity=None):
        self.file = h5py.File(hdf5_path, 'r')
        self.data = []
        self.image_names = image_names
        if self.image_names is not None:
            for img_n in self.image_names:
                image_np = np.array(self.file[img_n + '/image'])
                attributes = self.file[img_n].attrs['attributes']
                if specified_identity is None:
                    identity = int(self.file[img_n].attrs['identity'])
                else:
                    identity = specified_identity
                self.data.append((image_np, attributes, identity))
        self.img_transform = img_transform
        self.attr_transform = attr_transform

    def __getitem__(self, item):
        image, attributes, identity = self.data[item]
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.attr_transform is not None:
            attributes = self.attr_transform(attributes)
        return image, attributes, identity

    def __len__(self):
        return len(self.data)

# def load_celeba_multi_process():
#     import concurrent.futures
#     data_dir = '/home/tdye/VL/data/celeba'  # 存储分割数据的文件夹路径
#     full_data_dict = {}
#
#     def load_data(file_path):
#         with open(file_path, 'rb') as f:
#             part_data = pickle.load(f)
#         return part_data
#
#     files = [f_n for f_n in os.listdir(data_dir) if 'data_part' in f_n]
#     print(f"Loading {len(files)} data parts of CelebA.")
#     file_paths = [os.path.join(data_dir, f_n) for f_n in files]
#
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = executor.map(load_data, file_paths)
#
#     for result in results:
#         full_data_dict.update(result)
#     print(f"Size={len(full_data_dict.keys())}")
#     return full_data_dict

def get_property_classification_dataloaders(batch_size=16, property_id=0):
    train_loaders, test_loaders = [], []
    # loaded_data_dict = load_celeba()
    for client in train_clients:
        train_img_names = train_dataset[client]['x']
        test_img_names = test_dataset[client]['x']
        train_d = CelebA_DATASET(image_names=train_img_names,
                                 attr_transform=lambda x: torch.tensor(x[property_id]).long())
        test_d = CelebA_DATASET(image_names=test_img_names,
                                attr_transform=lambda x: torch.tensor(x[property_id]).long())
        c_train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        c_test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        train_loaders.append(c_train_loader)
        test_loaders.append(c_test_loader)
    clients = [i for i in range(len(train_clients))]
    return clients, train_loaders, test_loaders


def get_pearson_correlation_between_properties(property_id_A=20, property_id_B=31): # 20 Male, 31 Smiling
    imgs = []
    for client in train_clients:
        train_img_names = train_dataset[client]['x']
        imgs.extend(train_img_names)
    dataset = CelebA_DATASET(image_names=imgs,
                             attr_transform=lambda x: torch.tensor(x[0]).long())
    print(f"Len of all client training dataset: {len(dataset.data)}")
    p_A_values = [item[1][property_id_A] for item in dataset.data]
    p_B_values = [item[1][property_id_B] for item in dataset.data]

    # 创建两个特征的示例数据
    feature1 = np.array(p_A_values)
    feature2 = np.array(p_B_values)

    # 计算皮尔逊相关系数
    correlation_matrix = np.corrcoef(feature1, feature2)
    correlation_coefficient = correlation_matrix[0, 1]

    print(f"Pearson correlation coefficient: {correlation_coefficient}")

    return correlation_coefficient

def get_identity_classification_dataloaders(batch_size=10):
    train_loaders, test_loaders = [], []
    user_id = 0
    for client in train_clients:
        train_img_names = train_dataset[client]['x']
        test_img_names = test_dataset[client]['x']
        train_d = CelebA_DATASET(image_names=train_img_names,
                                 specified_identity=user_id)
        test_d = CelebA_DATASET(image_names=test_img_names,
                                specified_identity=user_id)
        c_train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        c_test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        train_loaders.append(c_train_loader)
        test_loaders.append(c_test_loader)
        user_id += 1
    clients = [i for i in range(len(train_clients))]
    return clients, train_loaders, test_loaders


# def get_celeba_combined_dataloader(batch_size=10, transform=None):
#     setup_seed(rs=42)
#     train_all_img_names = []
#     train_all_y = []
#     test_all_img_names = []
#     test_all_y = []
#     for client in train_clients:
#         c_train_img_names = train_dataset[client]['x']
#         train_all_img_names.extend(c_train_img_names)
#         c_train_y = train_dataset[client]['y']
#         train_all_y.extend(c_train_y)
#         c_test_img_names = test_dataset[client]['x']
#         test_all_img_names.extend(c_test_img_names)
#         c_test_y = test_dataset[client]['y']
#         test_all_y.extend(c_test_y)
#
#     train_d = CelebA_DATASET(img_names=train_all_img_names, labels=train_all_y, transform=transform)
#     test_d = CelebA_DATASET(img_names=test_all_img_names, labels=test_all_y, transform=transform)
#     train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
#     test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
#
#     return train_loader, test_loader

def get_property_classification_auxiliary_dataloader(num_train_samples=200, # 100 train/100 valid
                                                     num_test_samples=10000,
                                                     batch_size=16,
                                                     target_property_id=1,
                                                     random_state=42):
    print(f"Target task: property classification, property: {PID_2_NAME[target_property_id]}({target_property_id+1})")
    prop_nonprop_img_names = construct_prop_nonprop_datasets(property_id=target_property_id)
    prop_img_names = np.asarray(prop_nonprop_img_names['prop'])
    nonprop_img_names = np.asarray(prop_nonprop_img_names['nonprop'])
    # randomly select num_samples/2 prop and num_samples/2 negative
    prop_indices = np.random.choice(range(len(prop_img_names)), (num_train_samples + num_test_samples) // 2, replace=False)
    nonprop_indices = np.random.choice(range(len(nonprop_img_names)), (num_train_samples + num_test_samples) // 2, replace=False)
    prop_img_names = prop_img_names[prop_indices]
    nonprop_img_names = nonprop_img_names[nonprop_indices]
    img_names = np.hstack((prop_img_names, nonprop_img_names))
    labels = np.hstack(
        (np.ones_like(prop_img_names, dtype=np.int32),
         np.zeros_like(nonprop_img_names, dtype=np.int32)))
    # train test split, num_train_samples: num_test_samples (few-shot setting)
    train_img_names, test_img_names, train_labels, test_labels = train_test_split(img_names,
                                                                                  labels,
                                                                                  test_size=num_test_samples/(num_train_samples + num_test_samples),
                                                                                  random_state=random_state,
                                                                                  stratify=labels)

    train_img_names, valid_img_names, train_labels, valid_labels = train_test_split(train_img_names,
                                                                                  train_labels,
                                                                                  test_size=0.5,
                                                                                  random_state=random_state,
                                                                                  stratify=train_labels)

    train_d = CelebA_DATASET(image_names=train_img_names,
                             attr_transform=lambda x: torch.tensor(x[target_property_id]).long())
    valid_d = CelebA_DATASET(image_names=valid_img_names,
                             attr_transform=lambda x: torch.tensor(x[target_property_id]).long())
    test_d = CelebA_DATASET(image_names=test_img_names,
                            attr_transform=lambda x: torch.tensor(x[target_property_id]).long())

    train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, valid_loader, test_loader


def get_balanced_two_properties_classification_auxiliary_dataloader(
        num_train_samples=200, # 100 train/100 valid
        num_test_samples=10000,
        batch_size=16,
        main_property_id=35,
        target_property_id=15,
        random_state=42):
    print(f"Main task: property classification, property: {PID_2_NAME[main_property_id]}({main_property_id+1})")
    print(f"Target task: property classification, property: {PID_2_NAME[target_property_id]}({target_property_id+1})")
    main_prop_nonprop_img_names = construct_prop_nonprop_datasets(property_id=main_property_id)
    target_prop_nonprop_img_names = construct_prop_nonprop_datasets(property_id=target_property_id)

    main_prop_img_names = main_prop_nonprop_img_names['prop']
    main_nonprop_img_names = main_prop_nonprop_img_names['nonprop']

    target_prop_img_names = target_prop_nonprop_img_names['prop']
    target_nonprop_img_names = target_prop_nonprop_img_names['nonprop']

    main_prop_target_prop_img_names = np.array(list(set(main_prop_img_names) & set(target_prop_img_names)))
    main_prop_target_nonprop_img_names = np.array(list(set(main_prop_img_names) & set(target_nonprop_img_names)))

    main_nonprop_target_prop_img_names = np.array(list(set(main_nonprop_img_names) & set(target_prop_img_names)))
    main_nonprop_target_nonprop_img_names = np.array(list(set(main_nonprop_img_names) & set(target_nonprop_img_names)))

    # randomly select num_samples/4 per section
    prop_prop_indices = np.random.choice(range(len(main_prop_target_prop_img_names)), (num_train_samples + num_test_samples) // 4, replace=False)
    prop_nonprop_indices = np.random.choice(range(len(main_prop_target_nonprop_img_names)), (num_train_samples + num_test_samples) // 4, replace=False)
    nonprop_prop_indices = np.random.choice(range(len(main_nonprop_target_prop_img_names)), (num_train_samples + num_test_samples) // 4, replace=False)
    nonprop_nonprop_indices = np.random.choice(range(len(main_nonprop_target_nonprop_img_names)), (num_train_samples + num_test_samples) // 4, replace=False)

    prop_prop_img_names = main_prop_target_prop_img_names[prop_prop_indices]
    prop_nonprop_img_names = main_prop_target_nonprop_img_names[prop_nonprop_indices]
    nonprop_prop_img_names = main_nonprop_target_prop_img_names[nonprop_prop_indices]
    nonprop_nonprop_img_names = main_nonprop_target_nonprop_img_names[nonprop_nonprop_indices]

    img_names = np.hstack((prop_prop_img_names, prop_nonprop_img_names, nonprop_prop_img_names, nonprop_nonprop_img_names))
    labels = np.hstack(
        (np.ones_like(prop_prop_img_names, dtype=np.int32),
         np.ones_like(prop_nonprop_img_names, dtype=np.int32),
         np.zeros_like(nonprop_prop_img_names, dtype=np.int32),
         np.zeros_like(nonprop_nonprop_img_names, dtype=np.int32))
    )
    # train test split, num_train_samples: num_test_samples (few-shot setting)
    train_img_names, test_img_names, train_labels, test_labels = train_test_split(img_names,
                                                                                  labels,
                                                                                  test_size=num_test_samples/(num_train_samples + num_test_samples),
                                                                                  random_state=random_state,
                                                                                  stratify=labels)

    train_d = CelebA_DATASET(image_names=train_img_names,
                             attr_transform=lambda x: torch.tensor(x[main_property_id]).long())
    test_d = CelebA_DATASET(image_names=test_img_names,
                            attr_transform=lambda x: torch.tensor(x[main_property_id]).long())

    train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def get_face_recognition_auxiliary_dataloader(num_entity=100,
                                              ratio_of_test=0.6,
                                              batch_size=16,
                                              random_state=42):
    all_data_path = os.path.dirname(os.path.abspath(__file__)) + '/all'
    all_clients, all_dataset = read_dir(all_data_path)
    # X = []
    # Y = []
    candidate_clients = []
    for c in all_clients:
        if c not in train_clients and len(all_dataset[c]['y']) >= 30:
            candidate_clients.append(c)
    random.shuffle(candidate_clients)
    selected_clients = candidate_clients[:num_entity]
    user_id = 0
    public_data = []
    for c in selected_clients:
        c_dataset = CelebA_DATASET(image_names=all_dataset[c]['x'],
                                   specified_identity=user_id)
        public_data.extend(c_dataset.data)
        user_id += 1
    print(f"Total entity: {user_id}", f"total samples in public dataset :{len(public_data)}.")
    identities = [item[2] for item in public_data]
    public_train_data, public_test_data, train_identities, test_identities = train_test_split(public_data,
                                                                                  identities,
                                                                                  test_size=ratio_of_test,
                                                                                  random_state=random_state,
                                                                                  stratify=identities)
    public_train_data, public_valid_data, train_identities, valid_identities = train_test_split(public_train_data,
                                                                                  train_identities,
                                                                                  test_size=0.5,
                                                                                  random_state=random_state,
                                                                                  stratify=train_identities)
    train_d = CelebA_DATASET()
    train_d.data = public_train_data
    valid_d = CelebA_DATASET()
    valid_d.data = public_valid_data
    test_d = CelebA_DATASET()
    test_d.data = public_test_data

    train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_d, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    get_balanced_two_properties_classification_auxiliary_dataloader()
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
