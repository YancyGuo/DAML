# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:34:25 2016

@author: sakurai
"""

import os
import shutil
import tarfile
import subprocess
import numpy as np
from scipy.io import loadmat
import h5py
import fuel
from fuel.datasets.hdf5 import H5PYDataset
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    dataset_name = "prostate"
    archive_basename = "prostate_ims"

    fuel_root_path = "/home/ZSL/workspace/HDML/datasets/data"
    fuel_data_path = os.path.join(fuel_root_path, dataset_name)

    # 压缩数据文件路径 './datasets/data/cars196/car_ims.tgz'
    image_filepath = os.path.join(fuel_data_path, archive_basename)
    # 标签文件路径  './datasets/data/cars196/cars_annos.mat'
    label_filepath = os.path.join(fuel_data_path, "prostate_annos.mat")
    new_image_filepath = os.path.join(fuel_data_path, 'splited')

    # 根据test字段划分训练、测试集,并按序记录label
    prostate_annos = loadmat(label_filepath)  # 读取mat格式的标注信息
    annotations = prostate_annos["annotation_zsl"].ravel()  # 取"annotations"表的信息
    annotations = sorted(annotations, key=lambda a: str(a[0][0]))  # 按字段索引对annotations排序

    test_class_labels = []
    test_filenames = []

    train_filenames = []
    train_class_labels = []
    for annotation in annotations:
        test = annotation[2]
        if test:
            test_filenames.append(str(annotation[0][0]))
            class_label = int(annotation[1])  # 字段5，即"class"字段
            test_class_labels.append(class_label)
        else:
            train_filenames.append(str(annotation[0][0]))
            class_label = int(annotation[1])  # 字段5，即"class"字段
            train_class_labels.append(class_label)

    # filenames = train_filenames + test_filenames
    # class_labels = train_class_labels + test_class_labels
    # num_examples = len(filenames)
    #
    # # open hdf5 file
    # hdf5_filename = "prostate.hdf5"
    # hdf5_filepath = os.path.join(fuel_data_path, hdf5_filename)
    # hdf5 = h5py.File(hdf5_filepath, mode="w")  # 在hdf5_filepath路径下创建hdf5文件
    #
    # # store images
    # image_size = (256, 256)
    # array_shape = (num_examples, 3) + image_size
    # # 创建数据集images，shape为(num,3,h,w)
    # ds_images = hdf5.create_dataset("images", array_shape, dtype=np.uint8)
    # # 为每个数据集每个维度加标签
    # ds_images.dims[0].label = "batch"
    # ds_images.dims[1].label = "channel"
    # ds_images.dims[2].label = "height"
    # ds_images.dims[3].label = "width"

    # # # write images to the disk  ---------------每个病人读指定张数
    # prior = ''
    # num_per_patient = 0
    # for i, filename in tqdm(enumerate(train_filenames), total=len(train_filenames)):
    #     cur_patient_id = filename.split('_')[1]
    #     # 读取未处理图像
    #     if cur_patient_id == prior:
    #         num_per_patient += 1
    #     else:
    #         num_per_patient = 0
    #     if num_per_patient <= 2:
    #         image_path = os.path.join(image_filepath, filename)
    #         dst_path = os.path.join(new_image_filepath, 'train', filename)
    #         # raw_image = cv2.imread(image_path,
    #         #                        cv2.IMREAD_COLOR)  # BGR image
    #         shutil.copyfile(image_path, dst_path)
    #
    #     prior = cur_patient_id

    num_per_patient = 0
    for i, filename in tqdm(enumerate(train_filenames), total=len(train_filenames)):
        cur_patient_id = filename.split('_')[1]
        # 读取未处理图像
        # if not filename.split('_')[1] == prior:
        image_path = os.path.join(image_filepath, filename)
        dst_path = os.path.join(new_image_filepath, 'train', filename)
        # raw_image = cv2.imread(image_path,
        #                        cv2.IMREAD_COLOR)  # BGR image
        shutil.copyfile(image_path, dst_path)
        
        prior = filename.split('_')[1]

    for i, filename in tqdm(enumerate(test_filenames), total=len(test_filenames)):
        # 读取未处理图像
        image_path = os.path.join(image_filepath, filename)
        dst_path = os.path.join(new_image_filepath, 'test', filename)
        # raw_image = cv2.imread(image_path,
        #                        cv2.IMREAD_COLOR)  # BGR image
        shutil.copyfile(image_path, dst_path)


