# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:34:25 2016

@author: sakurai
"""

import os
import tarfile
import subprocess
import numpy as np
from scipy.io import loadmat
import h5py
import fuel
from fuel.datasets.hdf5 import H5PYDataset
import cv2
from tqdm import tqdm


def preprocess(hwc_bgr_image, size):
    """
    将HxWxC的BGR图片转换为CxHxW的RGB图像并缩放
    :param hwc_bgr_image:
    :param size:
    :return:
    """
    hwc_rgb_image = cv2.cvtColor(hwc_bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(hwc_rgb_image, (size))
    chw_image = np.transpose(resized, axes=(2, 0, 1))
    return chw_image


if __name__ == '__main__':
    dataset_name = "prostate"
    archive_basename = "prostate_ims"

    # 保存hdf5文件路径
    fuel_root_path = "/bigdata/ZSL/workSpace/tensorflow/HDML/datasets/data"
    fuel_data_path = os.path.join(fuel_root_path, dataset_name)
    hdf5_filename = "prostate_2_class_aug5times.hdf5"

    # 压缩数据文件路径 './datasets/data/cars196/car_ims.tgz'
    train_image_filepath = '/bigdata/ZSL/workSpace/tensorflow/HDML/datasets/split_by_patients/train'
    test_image_filepath = '/bigdata/ZSL/workSpace/tensorflow/HDML/datasets/split_by_patients/test'
    # querySet_filepath = '/home/ZSL/workspace/HDML/datasets/data/prostate/splited/queryset'

    # # 标签文件路径  './datasets/data/cars196/cars_annos.mat'
    # label_filepath = os.path.join(fuel_data_path, "prostate_annos.mat")

    # # 根据test字段划分训练、测试集,并按序记录label
    # prostate_annos = loadmat(label_filepath)  # 读取mat格式的标注信息
    # annotations = prostate_annos["annotation_zsl"].ravel()  # 取"annotations"表的信息
    # annotations = sorted(annotations, key=lambda a: str(a[0][0]))  # 按字段索引对annotations排序

    test_class_labels = []
    # test_filenames = []

    # train_filenames = []
    train_class_labels = []
    # querySet_class_labels = []

    train_filenames = os.listdir(train_image_filepath)
    test_filenames = os.listdir(test_image_filepath)
    # query_filenames = os.listdir(querySet_filepath)
    for train_filename in train_filenames:
        class_label = train_filename.split('_')[0]
        if not class_label == '0':
            class_label = 1
        train_class_labels.append(class_label)

    for test_filename in test_filenames:
        class_label = test_filename.split('_')[0]
        if not class_label == '0':
            class_label = 1
        test_class_labels.append(class_label)

    # for query_filename in query_filenames:
    #     class_label = query_filename.split('_')[0]
    #     if not class_label == '0':
    #         class_label = 1
    #     querySet_class_labels.append(class_label)


    # filenames = train_filenames + test_filenames
    # class_labels = train_class_labels + test_class_labels + querySet_class_labels
    # num_examples = len(train_filenames) + len(test_filenames) + len(query_filenames)

    class_labels = train_class_labels + test_class_labels
    num_examples = len(train_filenames) + len(test_filenames)

    # open hdf5 file
    hdf5_filepath = os.path.join(fuel_data_path, hdf5_filename)
    hdf5 = h5py.File(hdf5_filepath, mode="w")  # 在hdf5_filepath路径下创建hdf5文件

    # store images
    image_size = (227, 227)
    array_shape = (num_examples, 3) + image_size
    # 创建数据集images，shape为(num,3,h,w)
    ds_images = hdf5.create_dataset("images", array_shape, dtype=np.uint8)
    # 为每个数据集每个维度加标签
    ds_images.dims[0].label = "batch"
    ds_images.dims[1].label = "channel"
    ds_images.dims[2].label = "height"
    ds_images.dims[3].label = "width"

    # write images to the disk
    for i, filename in tqdm(enumerate(train_filenames), total=len(train_filenames),
                            desc=hdf5_filepath):
        # 读取未处理图像
        image_path = os.path.join(train_image_filepath, filename)
        raw_image = cv2.imread(image_path,
                               cv2.IMREAD_COLOR)  # BGR image
        # 将HxWxC的BGR图片转换为CxHxW的RGB图像并缩放
        image = preprocess(raw_image, image_size)
        ds_images[i] = image

    # write images to the disk
    for i, filename in tqdm(enumerate(test_filenames), total=len(test_filenames),
                            desc=hdf5_filepath):
        k = len(train_filenames) + i
        # 读取未处理图像
        image_path = os.path.join(test_image_filepath, filename)
        raw_image = cv2.imread(image_path,
                               cv2.IMREAD_COLOR)  # BGR image
        # 将HxWxC的BGR图片转换为CxHxW的RGB图像并缩放
        image = preprocess(raw_image, image_size)
        ds_images[k] = image

    # # write images to the disk
    # for i, filename in tqdm(enumerate(query_filenames), total=len(query_filenames),
    #                         desc=hdf5_filepath):
    #     start_idx = len(train_filenames) + len(test_filenames) + i
    #     # 读取未处理图像
    #     image_path = os.path.join(querySet_filepath, filename)
    #     raw_image = cv2.imread(image_path,
    #                             cv2.IMREAD_COLOR)  # BGR image
    #     # 将HxWxC的BGR图片转换为CxHxW的RGB图像并缩放
    #     image = preprocess(raw_image, image_size)
    #     ds_images[start_idx] = image

    # store the targets (class labels)
    # 创建数据集targets，shape为(num,1)
    targets = np.array(class_labels, np.int32).reshape(num_examples, 1)
    ds_targets = hdf5.create_dataset("targets", data=targets)
    ds_targets.dims[0].label = "batch"
    ds_targets.dims[1].label = "class_labels"

    # specify the splits (labels 1~98 for train, 99~196 for test)
    # 划分训练集、测试集
    test_head = len(train_filenames)  # 1137

    split_train, split_test = (0, test_head), (test_head, num_examples)
    # query_head = test_head + len(test_filenames)
    # split_train, split_test, split_query = (0, test_head), (test_head, query_head), (query_head, num_examples)

    split_dict = dict(train=dict(images=split_train, targets=split_train),
                      test=dict(images=split_test, targets=split_test))
    # split_dict = dict(train=dict(images=split_train, targets=split_train),
    #                   test=dict(images=split_test, targets=split_test),
    #                   query=dict(images=split_query, targets=split_query))
    hdf5.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    hdf5.flush()
    hdf5.close()
