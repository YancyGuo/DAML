# -*- coding: utf-8 -*-
import collections
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, BatchSizeScheme, SequentialScheme

from flags.FLAGS import FLAGS
from datasets.cars196_dataset import Cars196Dataset
from datasets.cub200_2011_dataset import Cub200_2011Dataset
from datasets.online_products_dataset import OnlineProductsDataset
from datasets.random_fixed_size_crop_mod import RandomFixedSizeCrop
from datasets.prostate_dataset import ProstateDataset

import random


def get_streams(batch_size=50, dataset='cars196', method='n_pairs_mc',
                crop_size=224, load_in_memory=False):
    """
    生成符合网络要求的数据流
    args:
    args:
        batch_size (int):
            number of examples per batch
        dataset (str):
            specify the dataset from 'cars196', 'cub200_2011', 'products'.
        method (str or fuel.schemes.IterationScheme):
            batch construction method. Specify 'n_pairs_mc', 'clustering', or
            a subclass of IterationScheme that has constructor such as
            `__init__(self, batch_size, dataset_train)` .
        crop_size (int or tuple of ints):
            height and width of the cropped image.
    """
    if dataset == 'cars196':
        dataset_class = Cars196Dataset
    elif dataset == 'cub200_2011':
        dataset_class = Cub200_2011Dataset
    elif dataset == 'products':
        dataset_class = OnlineProductsDataset
    elif dataset == 'prostate':
        dataset_class = ProstateDataset
    else:
        raise ValueError(
            "`dataset` must be 'cars196', 'cub200_2011', 'products' or 'prostate'.")

    dataset_train = dataset_class(['train'], load_in_memory=load_in_memory)
    # dataset_val = dataset_class(['val'], load_in_memory=load_in_memory)
    dataset_test = dataset_class(['test'], load_in_memory=load_in_memory)
    # dataset_query = dataset_class(['query'], load_in_memory=load_in_memory)

    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)

    # 按各度量学习方法生成数据对
    if method == 'n_pairs_mc':
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        # scheme = NPairLossScheme_no_random(labels, batch_size)  # 按顺序采样
        scheme = NPairLossScheme(labels, batch_size)
    elif method == 'clustering':
        scheme = EpochwiseShuffledInfiniteScheme(
            dataset_train.num_examples, batch_size)
    elif method == 'Contrastive_Loss':
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        scheme = ContrastiveLossScheme(labels, batch_size)
    elif method == 'Triplet':
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        scheme = TripletLossScheme(labels, batch_size)
    elif method == 'easy_pos_hard_negLoss':
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        # scheme = NPairLossScheme_no_random(labels, batch_size)  # 按顺序采样
        scheme = EPHN_LossScheme(labels, batch_size)
        # scheme = EPHN_LossScheme_no_random(labels, batch_size)
        # scheme = TripletLossScheme(labels, batch_size)   # 改成采a\p\n
    elif method == 'cross_entropy':
        labels = dataset_class(
            ['train'], sources=['targets'], load_in_memory=True).data_sources
        scheme = CE_LossScheme(labels, batch_size)
    elif issubclass(method, IterationScheme):
        scheme = method(batch_size, dataset=dataset_train)
    else:
        raise ValueError("`method` must be 'n_pairs_mc' or 'clustering' "
                         "or subclass of IterationScheme.")


    # 训练集  这个对象是fuel的stream
    stream = DataStream(dataset_train, iteration_scheme=scheme)  # 来自数据集的数据流....只用这个，不做增强，会出现nan！！！！！
    # 随机裁剪图像到一个固定的窗口大小。  这个是dataset对象
    stream_train = RandomFixedSizeCrop(stream, which_sources=('images',),
                                       random_lr_flip=False,  # 本来是true
                                       window_shape=crop_size)

    # # 验证集
    stream_train_eval = DataStream(
        dataset_train, iteration_scheme=SequentialScheme(
            dataset_train.num_examples, batch_size))
    # stream_train_eval = DataStream(   # 效果并不好
    #     dataset_query, iteration_scheme=SequentialScheme(
    #         dataset_query.num_examples, batch_size))

    # stream_train = DataStream(
    #     dataset_train, iteration_scheme=SequentialScheme(
    #         dataset_train.num_examples, batch_size))
    #
    # 测试集
    # 原来的程序报错：ValueError: Cannot feed value of shape (14, 0, 0, 3)
    # for Tensor 'Placeholder:0', which has shape '(?, 227, 227, 3)'
    # stream_test = RandomFixedSizeCrop(DataStream(
    #     dataset_test, iteration_scheme=SequentialScheme(
    #         dataset_test.num_examples, batch_size)),
    #     which_sources=('images',), center_crop=True, window_shape=crop_size)
    # # 测试集
    stream_test = DataStream(
        dataset_test, iteration_scheme=SequentialScheme(
            dataset_test.num_examples, batch_size))

    # # #####################################
    #         训练集 验证集、测试集          #
    ########################################
    # stream_train = DataStream(dataset_train, iteration_scheme=scheme)  # 来自数据集的数据流....只用这个，不做增强，会出现nan
    # stream_train_eval = DataStream(
    #     dataset_val, iteration_scheme=SequentialScheme(
    #         dataset_val.num_examples, batch_size))
    # stream_test = DataStream(
    #     dataset_test, iteration_scheme=SequentialScheme(
    #         dataset_test.num_examples, batch_size))
    print(stream, stream_train, stream_train_eval, stream_test)
    return stream, stream_train, stream_train_eval, stream_test


class NPairLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        # assert batch_size <= self.num_classes * 2, (
        #     "batch_size must not exceed twice the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        anchor_indexes, positive_indexes = self._generate_indexes()
        indexes = anchor_indexes + positive_indexes
        return indexes

    def _generate_indexes(self):
        # random_classes = np.random.choice(
        #     self.num_classes, self.batch_size // 2, False)
        random_classes = np.random.choice(
            self.num_classes, 2, False)
        anchor_indexes = []
        positive_indexes = []
        for c in random_classes:
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
            num = len(samples)
            anchor_indexes.extend(samples[:int(num/2)])
            positive_indexes.extend(samples[int(num/2):])
        return anchor_indexes, positive_indexes

    def get_request_iterator(self):
        return self


class EPHN_LossScheme(BatchSizeScheme):
    """每个类别batch_size/class张"""
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        # assert batch_size <= self.num_classes * 2, (
        #     "batch_size must not exceed twice the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        indexes = self._generate_indexes()
        return indexes

    def _generate_indexes(self):
        # random_classes = np.random.choice(
        #     self.num_classes, self.batch_size // 2, False)
        random_classes = np.random.choice(
            self.num_classes, 2, False)
        indexes = []
        for c in random_classes:
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
            indexes.extend(samples)
        return indexes

    def get_request_iterator(self):
        return self


class NPairLossScheme_no_random(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        self.iternum = 0
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        # assert batch_size <= self.num_classes * 2, (
        #     "batch_size must not exceed twice the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        anchor_indexes, positive_indexes = self._generate_indexes()
        indexes = anchor_indexes + positive_indexes
        self.iternum += 1
        print('iternum:', self.iternum)
        return indexes

    def _generate_indexes(self):
        # random_classes = np.random.choice(
        #     self.num_classes, self.batch_size // 2, False)
        # random_classes = np.random.choice(
        #     self.num_classes, 2, False)

        anchor_indexes = []
        positive_indexes = []
        num_per_class = FLAGS.batch_size // 2
        for c in range(0, self.num_classes):
            start = (self.iternum * num_per_class) % len(self._class_to_indexes[c])
            end = min(start + num_per_class, len(self._class_to_indexes[c]))
            print('\n', '...start:', start, '\n',
                  '...end:', end, '\n',
                  '...iternum:', self.iternum)
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            # samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
            samples = list(self._class_to_indexes[c][start: end])

            if not end-start == num_per_class:
                samples.extend(self._class_to_indexes[c][0: num_per_class-(end-start)])
            num = len(samples)
            anchor_indexes.extend(samples[:int(num/2)])
            positive_indexes.extend(samples[int(num/2):])

        print('\n', '...index1:', anchor_indexes, '\n',
                   '...index2:', positive_indexes)
        return anchor_indexes, positive_indexes

    def get_request_iterator(self):
        return self


class EPHN_LossScheme_no_random(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        self.iternum = 0
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        # assert batch_size <= self.num_classes * 2, (
        #     "batch_size must not exceed twice the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        indexes = self._generate_indexes()
        self.iternum += 1
        # print('iternum:', self.iternum)
        return indexes

    def _generate_indexes(self):
        # random_classes = np.random.choice(
        #     self.num_classes, self.batch_size // 2, False)
        # random_classes = np.random.choice(
        #     self.num_classes, 2, False)

        indexes = []
        num_per_class = FLAGS.batch_size // 2
        for c in range(0, self.num_classes):
            start = (self.iternum * num_per_class) % len(self._class_to_indexes[c])
            end = min(start + num_per_class, len(self._class_to_indexes[c]))
            # print('\n', '...start:', start, '\n',
            #       '...end:', end, '\n',
            #       '...iternum:', self.iternum)
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            # samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
            samples = list(self._class_to_indexes[c][start: end])


            if not end-start == num_per_class:
                samples.extend(self._class_to_indexes[c][0: num_per_class-(end-start)])
            indexes.extend(samples)

        # print("第", self.iternum,"轮样本：", indexes)
        return indexes

    def get_request_iterator(self):
        return self


class TripletLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()  # [0,0,0,...,1,1,1,...2,2,2,...,3,3,3,...]
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_  # [0 1 2 3]
        self.num_classes = len(self._classes)  # 4
        self.iternum = 0
        assert batch_size % 3 == 0, ("batch_size must be 3*n.")
        # assert batch_size <= self.num_classes * 3, (
        #     "batch_size must not exceed 3 times the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 3))
        self.batch_size = batch_size  # 18

        # 存储每个类别样本的下标 ，类别0的样本是【0~420】。。。
        self._class_to_indexes = []
        for c in self._classes:  # 遍历 _classs:[0 1 2 3]
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):  # 生成每一批数据划分好的正负锚样本
        # anchor_indexes, positive_indexes, negative_indexes = self._generate_indexes()
        # indexes = anchor_indexes + positive_indexes + negative_indexes
        indexes = self._generate_indexes()
        self.iternum += 1
        #        print indexes
        return indexes

    def _generate_indexes(self):
        # # random_classes = random.sample(
        # #     list(range(self.num_classes)), self.batch_size // 3 * 2)
        # random_classes = random.sample(  # <class 'list'>: [2,  3, 1, 0]
        #     list(range(self.num_classes)), 2)  # 选4类
        # anchor_indexes = []
        # positive_indexes = []
        # negative_indexes = []
        # # for i in range(self.batch_size // 3):
        #
        # # for i in range(2):
        #
        # # print(random_classes[i])  # 10
        # # print(self._class_to_indexes[random_classes[i]])]]]
        # samples = random.sample(list(self._class_to_indexes[random_classes[0]]), 8)  # 6
        # # a, p = random.sample(list(self._class_to_indexes[random_classes[i]]), 2)
        # num = len(samples)
        # anchor_indexes.extend(samples[:int(num / 2)])
        # positive_indexes.extend(samples[int(num / 2):])
        # #            while
        # #            c2 = np.random.choice(self.num_classes, 1)
        # #            while c2[0]==c:
        # #                c2 = np.random.choice(self.num_classes, 1)
        # ##            n = random.sample(list(self._class_to_indexes[c2[0]]), 1)
        # #            print c2, n
        # # n = random.sample(list(
        # #     self._class_to_indexes[random_classes[i + self.batch_size // 3]]), 1)
        # n = random.sample(list(
        #     self._class_to_indexes[random_classes[1]]), 4)  # 3
        # negative_indexes.extend(n)
        #
        # #        print anchor_indexes, positive_indexes, negative_indexes
        # #        positive_indexe
        # # print('anchor_indexes:', anchor_indexes, '\n', 'positive_indexes:', positive_indexes, '\n', 'negative_indexes:', negative_indexes)

        # ---------随机采样--------------
        # anchor_indexes = random.sample(list(self._class_to_indexes[1]), 8)
        # positive_indexes = random.sample(list(self._class_to_indexes[1]), 8)
        # negative_indexes = random.sample(list(self._class_to_indexes[0]), 8)

        random_classes = np.random.choice(
            self.num_classes, 2, False)
        indexes = []
        for i in range(len(random_classes)):
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)

            a_and_p = random.sample(list(self._class_to_indexes[random_classes[i-1]]), FLAGS.batch_size // 3*2)
            n = random.sample(list(self._class_to_indexes[random_classes[i]]), FLAGS.batch_size // 3)
            indexes = a_and_p+n
        return indexes

        # indexes = []
        # num_per_class = FLAGS.batch_size // 3
        #
        # start1 = (self.iternum * num_per_class * 2) % len(self._class_to_indexes[1])  # 取两份
        # end1 = min(start1 + num_per_class, len(self._class_to_indexes[1]))
        # # print('\n', '...start:', start, '\n',
        # #       '...end:', end, '\n',
        # #       '...iternum:', self.iternum)
        # # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
        # # samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
        # samples = list(self._class_to_indexes[1][start1: end1])
        #
        # if not end1 - start1 == num_per_class:
        #     samples.extend(self._class_to_indexes[1][0: num_per_class - (end1 - start1)])
        # indexes.extend(samples)
        #
        # start2 = (self.iternum * num_per_class) % len(self._class_to_indexes[0])  # 取两份
        # end2 = min(start2 + num_per_class, len(self._class_to_indexes[0]))
        # # print('\n', '...start:', start, '\n',
        # #       '...end:', end, '\n',
        # #       '...iternum:', self.iternum)
        # # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
        # # samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
        # samples = list(self._class_to_indexes[0][start2: end2])
        #
        # if not end1 - start1 == num_per_class:
        #     samples.extend(self._class_to_indexes[0][0: num_per_class - (end2 - start2)])
        # indexes.extend(samples)

        # return indexes
        return anchor_indexes + positive_indexes + negative_indexes

    def get_request_iterator(self):
        return self


class ContrastiveLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 2 == 0, ("batch_size must be 2*n.")
        # assert batch_size <= self.num_classes * 2, (
        #     "batch_size must not exceed twice times the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        pos1, neg1 = self._generate_indexes()
        indexes = pos1 + neg1
        #        print indexes
        return indexes

    def _generate_indexes(self):
        # indexes = random.sample(range(len(self._labels)),self.batch_size)
        # random_classes = random.sample(
        #     list(range(self.num_classes)), self.num_classes)
        # pos1 = []
        # pos2 = []
        # neg1 = []
        # neg2 = []
        # for i in range(self.batch_size // 4):
        #     a, p = random.sample(list(self._class_to_indexes[random_classes[i]]), 2)
        #     pos1.append(a)
        #     pos2.append(p)
        # for i in range((self.batch_size // 4), (self.batch_size // 4) * 2):
        #     n1 = random.sample(list(
        #         self._class_to_indexes[random_classes[i]]), 1)
        #     n2 = random.sample(list(
        #         self._class_to_indexes[random_classes[i + self.batch_size // 4]]), 1)
        #     neg1.append(n1)
        #     neg2.append(n2)

        random_classes1 = random.sample(
            list(range(self.num_classes)), self.num_classes//2)
        random_classes2 = random.sample(
            list(range(self.num_classes)), self.num_classes // 2)
        # pos1 = []
        # pos2 = []
        # neg1 = []
        # neg2 = []
        # a, p = random.sample(list(self._class_to_indexes[random_classes[0]]), FLAGS.batch_size//2)
        # pos1.append(a)
        # pos2.append(p)
        # n1, n2 = random.sample(list(
        #     self._class_to_indexes[random_classes[1]]), FLAGS.batch_size//2)
        # neg1.append(n1)
        # neg2.append(n2)

        anchor_indexes = []
        positive_indexes = []

        # for i in self.num_classes:
        # # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
        # anchor_indexes = np.random.choice(self._class_to_indexes[random_classes1[0]], FLAGS.batch_size // 2, False)
        # positive_indexes = np.random.choice(self._class_to_indexes[random_classes2[0]], FLAGS.batch_size // 2, False)
        # print("randomclass1:", random_classes1, '...index1:', anchor_indexes, '\n', 'randomclass2:', random_classes2, '...index2:', positive_indexes)
        # # num = len(samples)
        # # anchor_indexes.extend(samples[:int(num / 2)])
        # # positive_indexes.extend(samples[int(num / 2):])

        # return anchor_indexes, positive_indexes

        random_classes = np.random.choice(
            self.num_classes, 2, False)
        orignal_anchor_indexes = []
        orignal_positive_indexes = []
        anchor_indexes = []
        positive_indexes = []
        num = 0
        for c in random_classes:
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size // 2, False)
            num = len(samples)
            orignal_anchor_indexes.extend(samples[:int(num / 2)])
            orignal_positive_indexes.extend(samples[int(num / 2):])

        tmp = orignal_positive_indexes[:int(FLAGS.batch_size//8)]
        orignal_positive_indexes[:int(FLAGS.batch_size//8)] = orignal_positive_indexes[int(2*FLAGS.batch_size//8): int(3*FLAGS.batch_size//8)]
        orignal_positive_indexes[int(2*FLAGS.batch_size//8): int(3*FLAGS.batch_size//8)] = tmp
        anchor_indexes.extend(orignal_anchor_indexes)
        positive_indexes.extend(orignal_positive_indexes)
        # for i in range(0, len(orignal_anchor_indexes)):
        #     for j in range(0, len(orignal_anchor_indexes)):
        #         anchor_indexes.append(orignal_anchor_indexes[i])
        #         positive_indexes.append(orignal_positive_indexes[j])
        # print("11111:", len(anchor_indexes))
        # print("22222:", len(positive_indexes))
        return anchor_indexes, positive_indexes

        '''
        matches = []
        for i in range(self.batch_size // 2):
            if anchor_indexes[i] == positive_indexes[i]:
                matches += [1]
            else:
                matches += [0]
        anchor_indexes = np.array(anchor_indexes)
        positive_indexes = np.array(positive_indexes)
        matches = np.array(matches)    
        '''

    #        print anchor_indexes, positive_indexes, negative_indexes
    #        positive_indexe
    def get_request_iterator(self):
        return self


class EpochwiseShuffledInfiniteScheme(BatchSizeScheme):
    def __init__(self, indexes, batch_size):
        if not isinstance(indexes, collections.Iterable):
            indexes = range(indexes)
        if batch_size > len(indexes):
            raise ValueError('batch_size must not be larger than the indexes.')
        if len(indexes) != len(np.unique(indexes)):
            raise ValueError('Items in indexes must be unique.')
        self._original_indexes = np.array(indexes, dtype=np.int).flatten()
        self.batch_size = batch_size
        self._shuffled_indexes = np.array([], dtype=np.int)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        batch_size = self.batch_size

        # if remaining indexes are shorter than batch_size then new shuffled
        # indexes are appended to the remains.
        num_remains = len(self._shuffled_indexes)
        if num_remains < batch_size:
            num_overrun = batch_size - num_remains
            new_shuffled_indexes = self._original_indexes.copy()

            # ensure the batch of indexes from the joint part does not contain
            # duplicate index.
            np.random.shuffle(new_shuffled_indexes)
            overrun = new_shuffled_indexes[:num_overrun]
            next_indexes = np.concatenate((self._shuffled_indexes, overrun))
            while len(next_indexes) != len(np.unique(next_indexes)):
                np.random.shuffle(new_shuffled_indexes)
                overrun = new_shuffled_indexes[:num_overrun]
                next_indexes = np.concatenate(
                    (self._shuffled_indexes, overrun))
            self._shuffled_indexes = np.concatenate(
                (self._shuffled_indexes, new_shuffled_indexes))
        next_indexes = self._shuffled_indexes[:batch_size]
        self._shuffled_indexes = self._shuffled_indexes[batch_size:]
        return next_indexes.tolist()

    def get_request_iterator(self):
        return self


class CE_LossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        self.iternum = 0
        assert batch_size % 2 == 0, ("batch_size must be 2*n.")
        # assert batch_size <= self.num_classes * 2, (
        #     "batch_size must not exceed twice times the number of classes"
        #     "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        indexes = self._generate_indexes()
        self.iternum += 1
        return indexes

    def _generate_indexes(self):
        # random_classes = np.random.choice(
        #     self.num_classes, 2, False)
        # indexes = []
        # for c in random_classes:
        #     # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
        #     samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size // 2, False)
        #     indexes.extend(samples)

        # 按顺序采样
        indexes = []
        num_per_class = FLAGS.batch_size // 2
        for c in range(0, self.num_classes):
            start = (self.iternum * num_per_class) % len(self._class_to_indexes[c])
            end = min(start + num_per_class, len(self._class_to_indexes[c]))
            # print('\n', '...start:', start, '\n',
            #       '...end:', end, '\n',
            #       '...iternum:', self.iternum)
            # a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            # samples = np.random.choice(self._class_to_indexes[c], FLAGS.batch_size//2, False)
            samples = list(self._class_to_indexes[c][start: end])

            if not end - start == num_per_class:
                samples.extend(self._class_to_indexes[c][0: num_per_class - (end - start)])
            indexes.extend(samples)
        return indexes

    #        print anchor_indexes, positive_indexes, negative_indexes
    #        positive_indexe
    def get_request_iterator(self):
        return self