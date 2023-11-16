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
    dataset_name = "cars196"
    archive_basename = "car_ims"

    fuel_root_path = "/home/ZSL/workspace/HDML/datasets/data"
    fuel_data_path = os.path.join(fuel_root_path, dataset_name)

    # 数据文件路径 './datasets/data/cars196/car_ims.tar.gz'
    image_filepath = os.path.join(fuel_data_path, archive_basename + ".tgz")
    # 标签文件路径  './datasets/data/cars196/cars_annos.mat'
    label_filepath = os.path.join(fuel_data_path, "cars_annos.mat")

    # Extract car_ims.tgz if car_ims directory does not exist
    # 如果car_ims目录不存在,提取car_ims.tgz
    with tarfile.open(image_filepath, "r") as tf:  # './datasets/data/cars196/car_ims.tar.gz'
        # 获取以.jpg结尾的文件名
        jpg_filenames = [fn for fn in tf.getnames() if fn.endswith(".jpg")]
    jpg_filenames.sort()  # 对文件名进行排序
    num_examples = len(jpg_filenames)  # 样本数量

    # 若不存在基本数据的路径 './datasets/data/cars196/car_ims， 即没有解压好的数据
    if not os.path.exists(os.path.join(fuel_data_path, archive_basename)):
        # 则开启一个子进程去执行解压命令，并且等待子进程结束才继续执行其他的
        # tar zxvf image_filepath -C fuel_data_path --force-local 解压到fuel_data_path
        subprocess.call(["tar", "zxvf", image_filepath.replace("\\", "/"),
                         "-C", fuel_data_path.replace("\\", "/"),
                         "--force-local"])

    # Extract class labels  提取类别标签
    cars_annos = loadmat(label_filepath)   # 读取mat格式的标注信息
    annotations = cars_annos["annotations"].ravel()  # 取"annotations"表的信息
    annotations = sorted(annotations, key=lambda a: str(a[0][0]))  # 按字段索引对annotations排序
    class_labels = []
    for annotation in annotations:
        class_label = int(annotation[5])  # 字段5，即"class"字段
        class_labels.append(class_label)

    # open hdf5 file
    hdf5_filename = "cars196.hdf5"
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
    for i, filename in tqdm(enumerate(jpg_filenames), total=num_examples,
                            desc=hdf5_filepath):
        # 读取未处理图像
        raw_image = cv2.imread(os.path.join(fuel_data_path, filename),
                               cv2.IMREAD_COLOR)  # BGR image
        # 将HxWxC的BGR图片转换为CxHxW的RGB图像并缩放
        image = preprocess(raw_image, image_size)
        ds_images[i] = image

    # store the targets (class labels)
    # 创建数据集targets，shape为(num,1)
    targets = np.array(class_labels, np.int32).reshape(num_examples, 1)
    ds_targets = hdf5.create_dataset("targets", data=targets)
    ds_targets.dims[0].label = "batch"
    ds_targets.dims[1].label = "class_labels"

    # specify the splits (labels 1~98 for train, 99~196 for test)
    # 划分训练集、测试集
    test_head = class_labels.index(99)  # 8054
    split_train, split_test = (0, test_head), (test_head, num_examples)
    split_dict = dict(train=dict(images=split_train, targets=split_train),
                      test=dict(images=split_test, targets=split_test))
    hdf5.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    hdf5.flush()
    hdf5.close()
