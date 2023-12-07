from tqdm import tqdm
import pickle
from constants import celeba_transform
from PIL import Image
import os
import numpy as np
import math
import h5py

data_dict = {}
parent_path = '/home/tdye/leaf/data/celeba/data/raw/'

def get_metadata():
    f_identities = open(os.path.join(
        parent_path, 'identity_CelebA.txt'), 'r')
    identities = f_identities.read().split('\n')

    f_attributes = open(os.path.join(
        parent_path, 'list_attr_celeba.txt'), 'r')
    attributes = f_attributes.read().split('\n')

    return identities, attributes

def get_image2identity(identities):
    dic = {}

    for line in identities:
        info = line.split()
        if len(info) < 2:
            continue
        image, celeb = info[0], info[1]
        dic.update({image: celeb})

    return dic


def _get_celebrities_by_image(identities):
    good_images = {}
    for c in identities:
        images = identities[c]
        for img in images:
            good_images[img] = c
    return good_images

def get_image2attrs(attributes):
    col_idxes = [i for i in range(40)] # 40 attrs

    dic = {}
    for line in tqdm(attributes[2:]):
        info = line.split()
        if len(info) == 0:
            continue

        image = info[0]
        attrs = []
        for col_idx in col_idxes:
            attrs.append((int(info[1:][col_idx]) + 1) / 2)

        dic.update({image: attrs})

    return dic

def main():
    identities, attributes = get_metadata()
    image2identity_dic = get_image2identity(identities)
    image2attr_dic = get_image2attrs(attributes)
    with h5py.File('/home/tdye/VL/data/celeba/data_with_labels.h5', 'w') as hf:
        for filename in tqdm(os.listdir(os.path.join(parent_path, 'img_align_celeba'))):
            img_path = os.path.join(parent_path, 'img_align_celeba', filename)
            img_np = np.array(Image.open(img_path).convert('RGB'))

            hf.create_dataset(filename + '/image', data=img_np)
            hf[filename].attrs['attributes'] = image2attr_dic[filename]
            hf[filename].attrs['identity'] = image2identity_dic[filename]

if __name__ == '__main__':
    # main()
    with h5py.File('/home/tdye/VL/data/celeba/data_with_labels.h5', 'r') as hf:
        for img_name in tqdm(hf):
            # 读取图像数据
            image_np = np.array(hf[img_name + '/image'])

            # 读取属性标签和人脸类别标签
            attributes = hf[img_name].attrs['attributes']
            identity = hf[img_name].attrs['identity']
