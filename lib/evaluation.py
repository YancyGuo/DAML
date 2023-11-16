import operator
import os

import cv2
import numpy as np
import math
from scipy.special import comb
from sklearn import cluster
from sklearn import neighbors
import copy
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from lib.visualized_emb import *
import matplotlib.pyplot as plt
from embedding_visualized.tsne_visualized import plot_tsne
import torch

from lib.visualized_emb import embedding_Visual

# PROJECTOR 需要的日志文件名和地址相关参数
LOG_DIR = '/bigdata/ZSL/workSpace/tensorflow/HDML/tensorboard_log/prostate/easy_pos_hard_negLoss/09-05-11-44不需要标签一致插值重建permute'
SPRITE_FILE = 'sprite.jpg'
META_FILE = 'metadata.tsv'
TENSOR_FILE = 'tensor.tsv'
TEST_META_FILE = 'testset_metadata.tsv'
TEST_TENSOR_FILE = 'testset_tensor.tsv'


# return nmi,f1; n_cluster = num of classes 
def evaluate_cluster(feats, labels, n_clusters):
    """
    A function that calculate the NMI as well as F1 of a given embedding
    :param feats: The feature (embedding)
    :param labels: The labels
    :param n_clusters: How many classes
    :return: The NMI and F1 score of the given embedding
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=1).fit(feats)
    centers = kmeans.cluster_centers_

    # k-nearest neighbors
    neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers, range(len(centers)))

    idx_in_centers = neigh.predict(feats)
    num = len(feats)
    d = np.zeros(num)
    for i in range(num):
        d[i] = np.linalg.norm(feats[i, :] - centers[idx_in_centers[i], :])

    labels_pred = np.zeros(num)
    for i in np.unique(idx_in_centers):
        index = np.where(idx_in_centers == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid
    nmi, f1 = compute_clutering_metric(labels, labels_pred)
    return nmi, f1


def evaluate_recall(features_test, features_train, labels_train, labels_test, neighbours):
    """
    A function that calculate the recall score of a embedding
    :param features: The 2-d array of the embedding <class 'tuple'>: (281, 256)
    :param labels: The 1-d array of the label  <class 'tuple'>: (281, 1)
    :param neighbours: A 1-d array contains X in Recall@X  <class 'list'>: [1, 2, 4, 8, 16, 32]
    :return: A 1-d array of the Recall@X
    """
    dims = features_test.shape  # （281,256）
    recalls = []
    D2 = distance_matrix(features_test, features_train)  # <class 'tuple'>: (281, 281)

    # set diagonal to very high number
    num = dims[0]  # 281
    D = np.sqrt(np.abs(D2))  # 欧氏距离
    for i in range(0, np.shape(neighbours)[0]):
        recall_i = compute_recall_at_K(D, neighbours[i], labels_train, labels_test, num)
        recalls.append(recall_i)
    print('done')
    return recalls


def evaluate_recall_according_trainset(features_test, features_train, labels_train, labels_test, neighbours):
    """
    A function that calculate the recall score of a embedding
    :param features: The 2-d array of the embedding <class 'tuple'>: (281, 256)
    :param labels: The 1-d array of the label  <class 'tuple'>: (281, 1)
    :param neighbours: A 1-d array contains X in Recall@X  <class 'list'>: [1, 2, 4, 8, 16, 32]
    :return: A 1-d array of the Recall@X
    """
    dims = features_test.shape  # （281,256）
    recalls_k = []
    num = dims[0]  # 281

    max_f1score = 0

    # 通过knn算法得预测类别
    for k in range(0, np.shape(neighbours)[0]):
        y_pred = []
        num_correct = 0
        for i in range(0, num):  # i是测试集中的第i个待预测embedding
            this_gt_class_idx = labels_test[i]
            # y_pred_cur = knn_classify(features_test[i], features_train, labels_train, neighbours[k])
            y_pred_cur = knn_classify(features_test[i], features_train, labels_train, neighbours[k])
            # y_pred_cur, sorteDistIndicies_label = knn_classify_vote(features_test[i], features_train, labels_train, neighbours[k])
            # y_pred_cur = weighted_classify(features_test[i], features_train, labels_train, neighbours[k])
            if y_pred_cur == this_gt_class_idx:  # 若与测试集金标准相符合，则预测正确的数量+1
                num_correct = num_correct + 1
            # else:  # 打印金标准以及邻居
            #     print("金标准：", this_gt_class_idx)
            #     print("邻居：", sorteDistIndicies_label)
            y_pred.extend(y_pred_cur)

        y_true = labels_test
        metric = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=3)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1score = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        print(metric)
        if f1score>max_f1score:
            max_f1score = f1score

        recall_k = float(num_correct) / float(num)
        print('num_correct:', num_correct)
        print('num:', num)
        print("K: %d, Recall: %.3f\n" % (neighbours[k], recall_k))
        recalls_k.append(recall_k)
    print('done')
    return recalls_k, recall, precision, f1score, max_f1score


def knn_classify(inX, dataSet, labels, k):
    """

    :param inX: <class 'tuple'>: (1024,)  待分类向量
    :param dataSet: 训练样本<class 'tuple'>: (1137, 1024)
    :param labels: 训练样本标签<class 'tuple'>: (1137, 1)
    :param k: k-近邻算法中的k
    :return:
    """
    dataSetSize = dataSet.shape[0]  # 训练样本数，1137
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile 表示inX在重复dataSetSize行，重复1列。为输入向量与各个样本求取欧式距离做准备。(1137,1024)
    sqDiddMat = diffMat ** 2  # diffMat是输入向量与我们训练样本每个点相减得到的，**2表示值的结果取平方。 (1137,512)
    sqDistances = sqDiddMat.sum(axis=1)  # 默认为axis=0，axis=1以后就是将一个矩阵的每一行向量相加 (1137,)
    # distances = sqDistances ** 0.5  # 对结果进行开平方，得到输入向量与每个训练样本中点的欧式距离
    sorteDistIndicies = sqDistances.argsort()  # 将距离结果按照从小到大排序获得索引值  (1137,)
    # classcount = {}  # 这是一个字典，key为类别，value为距离最小的前k个样本点里面为该类别的个数。
    # for i in range(k):
    #     voteIlabel = labels[sorteDistIndicies[i]]  # 获取距离最小的前k个样本点对应的label值
    #     classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1  # 如果之前的样本点label值与与现在的相同，则累计加1，否则，此次加1

    # inds = np.array(np.argsort(this_row))[0]  # 由小到大排序
    knn_inds = sorteDistIndicies[0:k]  # 前k个距离最近的样本下标
    knn_class_inds = [labels[i] for i in knn_inds]  # 前k个距离最近的样本类别
    # k个中频数最大的类别
    y_pred_cur = max(knn_class_inds, key=knn_class_inds.count)

    # sorteClassCount = sorted(classcount.items(), key=operator.itemgetter(1),
    #                          reverse=True)  # 针对calsscount获取对象的第1个域的值进行降序排序。也就是说根据类别的个数从大到小排序。
    # return sorteClassCount[0][0]  # 返回排序的字典的第一个元素的key，即分类后的类别
    return y_pred_cur


def knn_classify_vote(inX, dataSet, labels, k):
    """

    :param inX: <class 'tuple'>: (1024,)  待分类向量
    :param dataSet: 训练样本<class 'tuple'>: (1137, 1024)
    :param labels: 训练样本标签<class 'tuple'>: (1137, 1)
    :param k: k-近邻算法中的k
    :return:
    """
    # 训练样本数，1137
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile 表示inX在重复dataSetSize行，重复1列。为输入向量与各个样本求取欧式距离做准备。(1137,1024)
    # 欧氏距离
    sqDiddMat = diffMat ** 2  # diffMat是输入向量与我们训练样本每个点相减得到的，**2表示值的结果取平方。 (1137,512)
    sqDistances = sqDiddMat.sum(axis=1)  # 默认为axis=0，axis=1以后就是将一个矩阵的每一行向量相加 (1137,)
    # distances = sqDistances ** 0.5  # 对结果进行开平方，得到输入向量与每个训练样本中点的欧式距离


    # 固定半径
    # fix_radius = np.mean(sqDistances)
    # 不平衡率  n_major/n_min
    class_0_num = np.sum(np.where(labels, 0, 1))  # 等于0的元素置为1  421
    class_1_num = dataSetSize - class_0_num  # 716
    imbalance_rate = class_1_num / class_0_num  # 1.700712589073634

    # 等级，1,2，..., 1137
    R_xp = 1 + np.arange(dataSetSize)
    # 少数类分数
    # G_min = imbalance_rate*((dataSetSize-R_xp)/(dataSetSize-1-np.sqrt(imbalance_rate)))  # <class 'tuple'>: (1137,)  [1.70266723,1.7011684,1.69966958
    G_min = imbalance_rate*(dataSetSize-R_xp)/dataSetSize  # <class 'tuple'>: (1137,)  [1.70266723,1.7011684,1.69966958
    G_maj = -imbalance_rate**(R_xp/dataSetSize)  # <class 'tuple'>: (1137,)  [-1.00046717,-1.00093456, -1.00140216
    # RG = [G_min,G_maj]  # (2,2274)

    # 累计分数超过这个阈值则分类结束
    W = 0.7*imbalance_rate   # 1.2  1.17

    sorteDistIndicies = sqDistances.argsort()  # 将距离结果按照从小到大排序获得索引值  (1137,)
    # fix_R_indicies = np.where(sqDistances < fix_radius)[0]

    # 符合条件的样本下标
    # indicies = [id for id in sorteDistIndicies if id in fix_R_indicies]
    sorteDistIndicies_label = [labels[i][0] for i in sorteDistIndicies]  # 对应标签

    # 分值和
    score = 0
    for k, id in enumerate(sorteDistIndicies):   # k是等级
        label = labels[id][0]
        if label == 0:  # 少数类
            score = score + G_min[k]
        else:
            score = score + G_maj[k]

        if abs(score) > W:  #
            if score > 0:
                y_pred_cur = 0
            else:
                y_pred_cur = 1
            return [y_pred_cur], sorteDistIndicies_label

    # score0 = 0
    # score1 = 0
    # for k in sorteDistIndicies:
    #     if labels[k] == 0:  # 少数类
    #         score0 = score0 + G_min[k]
    #     else:
    #         score1 = score1 + G_maj[k]
    #
    #     if abs(score0) > W:
    #         y_pred_cur = 0
    #         return [y_pred_cur]
    #     if abs(score1) >W:
    #         y_pred_cur = 1
    #         return [y_pred_cur]


    # # 少数类分数
    # G_min = imbalance_rate# <class 'tuple'>: (1137,)  [1.70266723,1.7011684,1.69966958
    # G_maj = 1  # <class 'tuple'>: (1137,)  [1.00046717,-1.00093456, -1.00140216
    # # RG = [G_min,G_maj]  # (2,2274)
    #
    # # 累计分数超过这个阈值则分类结束
    # # W = 0.7 * imbalance_rate  # 1.2
    #
    # # sorteDistIndicies = sqDistances.argsort()  # 将距离结果按照从小到大排序获得索引值  (1137,)
    # fix_R_indicies = np.where(sqDistances < fix_radius)[0]
    #
    # # # 符合条件的样本下标
    # # indicies = [id for id in sorteDistIndicies if id in fix_R_indicies]
    # # fix_r_knn_class_label = [labels[i] for i in indicies]  # 对应标签

    # # 分值和
    # score_min = 0
    # score_maj = 0
    # for k in fix_R_indicies:
    #     if labels[k] == 0:  # 少数类
    #         score_min = score_min + G_min
    #     else:
    #         score_maj = score_maj + G_maj
    # if score_min > score_maj:  #
    #     y_pred_cur = 0
    # else:
    #     y_pred_cur = 1
    # return [y_pred_cur]



    # classcount = {}  # 这是一个字典，key为类别，value为距离最小的前k个样本点里面为该类别的个数。
    # for i in range(k):
    #     voteIlabel = labels[sorteDistIndicies[i]]  # 获取距离最小的前k个样本点对应的label值
    #     classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1  # 如果之前的样本点label值与与现在的相同，则累计加1，否则，此次加1

    # inds = np.array(np.argsort(this_row))[0]  # 由小到大排序
    # knn_inds = sorteDistIndicies[0:k]  # 前k个距离最近的样本下标
    # knn_class_inds = [labels[i] for i in knn_inds]  # 前k个距离最近的样本类别

    # # k个中频数最大的类别
    # y_pred_cur = max(knn_class_inds, key=knn_class_inds.count)

    # sorteClassCount = sorted(classcount.items(), key=operator.itemgetter(1),
    #                          reverse=True)  # 针对calsscount获取对象的第1个域的值进行降序排序。也就是说根据类别的个数从大到小排序。
    # return sorteClassCount[0][0]  # 返回排序的字典的第一个元素的key，即分类后的类别


def gaussian(dist, sigma=10.0):
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


### 加权KNN
def weighted_classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    diff = np.tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = np.array([sum(x) for x in sqdiff])
    dist = squareDist ** 0.5
    # print(input, dist[0], dist[1164])
    sortedDistIndex = np.argsort(dist)

    classCount = {}
    for i in range(k):  # 遍历k个样本
        index = sortedDistIndex[i]  # 获取距离最小的第i个样本点对应的label值
        voteLabel = label[index][0]  # 当前样本类别
        weight = gaussian(dist[index])  # 给距离权重
        # print(index, dist[index],weight)
        ## 这里不再是加一，而是权重*1
        classCount[voteLabel] = classCount.get(voteLabel, 0) + weight * 1

    maxCount = 0
    # print(classCount)
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    # return classes
    return [classes]

def knn_classify_mulsum(inX, dataSet, labels, k):
    """

    :param inX: 用于分类的输入向量
    :param dataSet: 训练样本集合
    :param labels: 标签向量
    :param k: k-近邻算法中的k
    :return:
    """
    dataSetSize = dataSet.shape[0]  # 训练样本数，1137
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile 表示inX在重复dataSetSize行，重复1列。为输入向量与各个样本求取欧式距离做准备。(1137,512)
    mat = np.multiply(dataSet,diffMat)
    mat_sum = np.sum(mat, axis=1)  # (1137,)
    # sqDiddMat = diffMat ** 2  # diffMat是输入向量与我们训练样本每个点相减得到的，**2表示值的结果取平方。 (1137,512)
    # sqDistances = sqDiddMat.sum(axis=1)  # 默认为axis=0，axis=1以后就是将一个矩阵的每一行向量相加 (1137,)
    # distances = sqDistances ** 0.5  # 对结果进行开平方，得到输入向量与每个训练样本中点的欧式距离
    sorteDistIndicies = mat_sum.argsort()  # 将距离结果按照从小到大排序获得索引值  (1137,)
    # classcount = {}  # 这是一个字典，key为类别，value为距离最小的前k个样本点里面为该类别的个数。
    # for i in range(k):
    #     voteIlabel = labels[sorteDistIndicies[i]]  # 获取距离最小的前k个样本点对应的label值
    #     classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1  # 如果之前的样本点label值与与现在的相同，则累计加1，否则，此次加1

    # inds = np.array(np.argsort(this_row))[0]  # 由小到大排序
    knn_inds = sorteDistIndicies[0:k]  # 前k个距离最近的样本下标
    knn_class_inds = [labels[i] for i in knn_inds]  # 前k个距离最近的样本类别
    # k个中频数最大的类别
    y_pred_cur = max(knn_class_inds, key=knn_class_inds.count)

    # sorteClassCount = sorted(classcount.items(), key=operator.itemgetter(1),
    #                          reverse=True)  # 针对calsscount获取对象的第1个域的值进行降序排序。也就是说根据类别的个数从大到小排序。
    # return sorteClassCount[0][0]  # 返回排序的字典的第一个元素的key，即分类后的类别
    return y_pred_cur

# def evaluate_recall(features, labels, neighbours):
#     """
#     A function that calculate the recall score of a embedding
#     :param features: The 2-d array of the embedding <class 'tuple'>: (281, 256)
#     :param labels: The 1-d array of the label  <class 'tuple'>: (281, 1)
#     :param neighbours: A 1-d array contains X in Recall@X  <class 'list'>: [1, 2, 4, 8, 16, 32]
#     :return: A 1-d array of the Recall@X
#     """
#     dims = features.shape  # （281,256）
#     recalls = []
#     D2 = distance_matrix(features)  # <class 'tuple'>: (281, 281)
#
#     # set diagonal to very high number
#     num = dims[0]  # 281
#     D = np.sqrt(np.abs(D2))  # 欧氏距离
#     for i in range(0, np.shape(neighbours)[0]):
#         recall_i = compute_recall_at_K(D, neighbours[i], labels, num)
#         recalls.append(recall_i)
#     print('done')
#     return recalls


def compute_clutering_metric(idx, item_ids):
    N = len(idx)

    # cluster centers
    centers = np.unique(idx)
    num_cluster = len(centers)
    # print('Number of clusters: #d\n' % num_cluster);

    # count the number of objects in each cluster
    count_cluster = np.zeros(num_cluster)
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0])

    # build a mapping from item_id to item index
    keys = np.unique(item_ids)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])

    # count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[item_ids[i]]
        count_item[index] = count_item[index] + 1

    # compute purity
    purity = 0
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((num_cluster, num_item))
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0]
        index_item = item_map[item_ids[i]]
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1

    # mutual information
    I = 0
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s

    # entropy
    H_cluster = 0
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F = (beta * beta + 1) * P * R / (beta * beta * P + R)

    return NMI, F


# def distance_matrix(X):
#     """
#     计算X中各个元素两两之间的距离
#     :param X:
#     :return:
#     """
#     X = np.matrix(X)  # （281,512）
#     m = X.shape[0]  # 元素数量 281
#     t = np.matrix(np.ones([m, 1]))  # （281,1）的全1矩阵
#
#     x = np.matrix(np.empty([m, 1]))  # （281,1）的随机数矩阵，该矩阵用于记录每个embedding各项的平方和
#     for i in range(0, m):
#         n = np.linalg.norm(X[i, :])  # 对第个embedding求l2范数，即平方和开根号 0.99999994
#         x[i] = n * n  # 平方
#     # tmp1 = x * np.transpose(t)  # (281,281) 得到的结果就是把x矩阵的每一个元素复制281次
#     # tmp2 = t * np.transpose(x)  # 是tmp1的转置吧
#     # XXT = X * np.transpose(X)
#     # D = tmp1 + tmp2 - 2 * XXT  # （281,281）
#     D = x * np.transpose(t) + t * np.transpose(x) - 2 * X * np.transpose(X)  # （281,281）  # (a-b)^2=(a^2 + b^2- 2ab)
#     return D


def distance_matrix(train_emb, test_emb):
    X = np.matrix(train_emb)  # (281, 512)
    Y = np.matrix(test_emb)  # (960, 512)

    # X = np.matrix(X)  # （281,512）
    mx = X.shape[0]  # 元素数量 281
    tx = np.matrix(np.ones([mx, 1]))  # （281,1）的全1矩阵
    x = np.matrix(np.empty([mx, 1]))  # （281,1）的随机数矩阵，该矩阵用于记录每个embedding各项的平方和

    my = Y.shape[0]
    ty = np.matrix(np.ones([my, 1]))
    y = np.matrix(np.empty([my, 1]))
    for i in range(0, mx):
        n = np.linalg.norm(X[i, :])  # 对第个embedding求l2范数，即平方和开根号 0.99999994
        x[i] = n * n  # 平方

    for i in range(0, my):
        n = np.linalg.norm(Y[i, :])  # 对第个embedding求l2范数，即平方和开根号 0.99999994
        y[i] = n * n  # 平方
    # tmp1 = x * np.transpose(t)  # (281,281) 得到的结果就是把x矩阵的每一个元素复制281次
    # tmp2 = t * np.transpose(x)  # 是tmp1的转置吧
    # XXT = X * np.transpose(X)
    # D = tmp1 + tmp2 - 2 * XXT  # （281,281）
    D = x * np.transpose(tx) + y * np.transpose(ty) - X * np.transpose(Y) - Y * np.transpose(
        X)  # # (a-b)^2=(a^2 + b^2- 2ab)
    return D


def compute_recall_at_K(D, K, class_ids_train, class_ids_test, num):
    """
    knn
    :param D: 测试集与训练集的距离矩阵
    :param K:
    :param class_ids: 测试集的类别金标准
    :param num: 测试数量
    :return:
    """
    num_correct = 0
    y_pred = []
    # 通过knn算法得预测类别
    class_ids_test = class_ids_test.reshape([class_ids_test.shape[0]])
    class_ids_train = class_ids_train.reshape([class_ids_train.shape[0]])
    for i in range(0, num):  # i是测试集中的第i个待预测embedding
        this_gt_class_idx = class_ids_test[i]
        this_row = D[i, :]  # 与训练集中各个嵌入的距离
        inds = np.array(np.argsort(this_row))[0]  # 由小到大排序
        knn_inds = inds[0:K]  # 前k个距离最近的样本下标
        knn_class_inds = [class_ids_train[i] for i in knn_inds]  # 前k个距离最近的样本类别
        # k个中频数最大的类别
        y_pred_cur = max(knn_class_inds, key=knn_class_inds.count)
        if y_pred_cur == this_gt_class_idx:  # 若与测试集金标准相符合，则预测正确的数量+1
            num_correct = num_correct + 1
        y_pred.append(y_pred_cur)
    y_true = class_ids_test
    metric = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=3)
    print(metric)
    recall = float(num_correct) / float(num)

    print('num_correct:', num_correct)
    print('num:', num)
    print("K: %d, Recall: %.3f\n" % (K, recall))
    return recall


# def compute_recall_at_K(D, K, class_ids, num):
#     num_correct = 0
#     y_pred = []
#     # 通过knn算法得预测类别
#     class_ids = class_ids.reshape([class_ids.shape[0]])
#     for i in range(0, num): # i是测试集中的第i个待预测embedding
#         this_gt_class_idx = class_ids[i]
#         this_row = D[i, :]  # 与
#         inds = np.array(np.argsort(this_row))[0]
#         knn_inds = inds[0:K]  # 前k个距离最近的样本下标
#         knn_class_inds = [class_ids[i] for i in knn_inds]  # 前k个距离最近的样本类别
#         # 频数最大的类别
#         y_pred_cur = max(knn_class_inds, key=knn_class_inds.count)
#         if y_pred_cur == this_gt_class_idx:
#             num_correct = num_correct + 1
#         y_pred.append(y_pred_cur)
#     y_true = class_ids
#     metric = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=3)
#     print(metric)
#     recall = float(num_correct)/float(num)
#
#     print('num_correct:', num_correct)
#     print('num:', num)
#     print("K: %d, Recall: %.3f\n" % (K, recall))
#     return recall


def Evaluation_reconstruct_img(recon_imgs, stream_test, stream_train, image_mean, sess, x_raw, label_raw, is_Training, embedding, num_class,
               neighb, visualized_emb=False):
    """

    :param recon_imgs:
    :param stream_test: 测试集数据流
    :param image_mean:
    :param sess:
    :param x_raw: 占位符
    :param label_raw:  占位符
    :param is_Training:占位符
    :param embedding:
    :param num_class:
    :param neighb:
    :return:
    """
    y_batches = []
    c_batches = []
    test_img = []
    trian_img = []
    for batch in copy.copy(stream_test.get_epoch_iterator()):
        x_batch_data_temp, c_batch_data = batch  #
        x_batch_data = np.transpose(x_batch_data_temp[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        # y_batch, _embedding_yp_img = sess.run([tf.nn.l2_normalize(embedding, dim=1), embedding_yp_img],
        #                    feed_dict={x_raw: x_batch_data,
        #                               label_raw: c_batch_data,
        #                               is_Training: False})

        # y_batch, _recon_imgs= sess.run([tf.nn.l2_normalize(embedding, dim=1), recon_imgs],
        #                                 feed_dict={x_raw: x_batch_data,
        #                                            label_raw: c_batch_data,
        #                                            is_Training: False})

        y_batch= sess.run([tf.nn.l2_normalize(embedding, dim=1)],
                                        feed_dict={x_raw: x_batch_data,
                                                   label_raw: c_batch_data,
                                                   is_Training: False})

        _recon_imgs = sess.run(recon_imgs,
                                        feed_dict={x_raw: x_batch_data,
                                                   label_raw: c_batch_data,
                                                   is_Training: False})

        # Find predictions of classes that are not in the dataset.
        y_batch_data = y_batch[0]
        y_batches.append(y_batch_data)
        c_batches.append(c_batch_data)

        if visualized_emb:
            for x in x_batch_data:
                xx = cv2.cvtColor(cv2.resize(x + image_mean[0], (32, 32)), cv2.COLOR_BGR2GRAY)  # <class 'tuple'>: (3, 32, 32)
                test_img.append(xx / 255.0)

    y_data_test = np.concatenate(y_batches)
    c_data_test = np.concatenate(c_batches)

    # # 可视化重建图像
    # plt.imshow(_recon_imgs[0])
    # plt.imsave('./reconstructimgs',_recon_imgs[0])

    # 训练集
    y_batches_train = []
    c_batches_train = []
    recon_imgs_trian = []
    # for batch in tqdm(copy.copy(stream_train.get_epoch_iterator())):
    id = 0
    for batch in copy.copy(stream_train.get_epoch_iterator()):
        x_batch_data_temp, c_batch_data = batch  #
        x_batch_data = np.transpose(x_batch_data_temp[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1)],
                           feed_dict={x_raw: x_batch_data,
                                      label_raw: c_batch_data,
                                      is_Training: False})
        _recon_img_train = sess.run(recon_imgs,
                                             feed_dict={x_raw: x_batch_data,
                                                        label_raw: c_batch_data,
                                                        is_Training: False})

        # Find predictions of classes that are not in the dataset.
        y_batch_data = y_batch[0]
        y_batches_train.append(y_batch_data)  # 向量
        c_batches_train.append(c_batch_data)
        # recon_imgs_trian(_recon_img_train)
        if visualized_emb:
            for x in x_batch_data:
                # xx = cv2.resize(x, (32, 32)).transpose(2, 0, 1)
                # trian_img.append((xx + image_mean[0].transpose(2, 0, 1)) / 255.0)
                xx = cv2.cvtColor(cv2.resize(x + image_mean[0], (32, 32)),
                                  cv2.COLOR_BGR2GRAY)  # <class 'tuple'>: (3, 32, 32)
                trian_img.append(xx / 255.0)
            for x in _recon_img_train:
                # xx = cv2.resize(x, (32, 32)).transpose(2, 0, 1)
                # trian_img.append((xx + image_mean[0].transpose(2, 0, 1)) / 255.0)
                plt.imsave('./visulized/'+str(id), x)
                id += 1


    y_data_t = np.concatenate(y_batches_train)
    c_data_t = np.concatenate(c_batches_train)

    if visualized_emb:
        # # 生成Sprite图像
        sprite_image = create_sprite_image(trian_img)

        # 放到日志目录下
        path_for_prostate_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
        plt.imsave(path_for_prostate_sprites, sprite_image, cmap='gray')
        plt.imshow(sprite_image, cmap='gray')
        # 标签文件写入
        path_for_prostate_metadata = os.path.join(LOG_DIR, META_FILE)
        path_for_prostate_tensor = os.path.join(LOG_DIR, TENSOR_FILE)
        with open(path_for_prostate_metadata, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(c_data_t):
                f.write("%d\t%d\n" % (index, label))
        with open(path_for_prostate_tensor, 'w') as f:
            for index, embedding in enumerate(y_data_t):
                f.write('\t'.join([str(x) for x in embedding]) + "\n")

        # 测试集 tensor.tsv和metadata.tsv文件生成
        path_for_prostate_metadata_testset = os.path.join(LOG_DIR, META_FILE)
        path_for_prostate_tensor = os.path.join(LOG_DIR, TENSOR_FILE)
        with open(path_for_prostate_metadata, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(c_data_t):
                f.write("%d\t%d\n" % (index, label))
        with open(path_for_prostate_tensor, 'w') as f:
            for index, embedding in enumerate(y_data_t):
                f.write('\t'.join([str(x) for x in embedding]) + "\n")


        # 生成可视化向量所需要的日志问价
        # embedding_Visual(y_data_t, 'trainset_embedding', LOG_DIR, META_FILE, SPRITE_FILE, TRAINING_STEPS=8000)
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/experiment')
        # writer.add_embedding(y_data_t, c_data_t,
        #                      torch.from_numpy(np.stack(test_img, axis=0)),
        #                      global_step=8000, tag='embedding/train')
        # writer.flush()

    n_clusters = num_class
    nmi, f1 = evaluate_cluster(y_data_test, c_data_test, n_clusters)

    recalls_zsl, recall, precision, f1_score, max_f1score = evaluate_recall_according_trainset(features_test=y_data_test, features_train=y_data_t, labels_train=c_data_t,
                                  labels_test=c_data_test, neighbours=neighb)

    # print(nmi)
    # print(f1)
    return nmi, f1, recalls_zsl, recall, precision, f1_score, max_f1score
    # return recalls_zsl, recall, precision, f1_score, max_f1score


def Evaluation(stream_test, stream_train, image_mean, sess, x_raw, label_raw, is_Training, embedding, num_class,
               neighb, visualized_emb=False, visualized_tsne=True, model_path=''):
    """

    :param stream_test: 测试集数据流
    :param image_mean:
    :param sess:
    :param x_raw: 占位符
    :param label_raw:  占位符
    :param is_Training:占位符
    :param embedding:
    :param num_class:
    :param neighb:
    :return:
    """
    y_batches = []
    c_batches = []
    test_img = []
    trian_img = []
    for batch in copy.copy(stream_test.get_epoch_iterator()):
        x_batch_data_temp, c_batch_data = batch  #
        x_batch_data = np.transpose(x_batch_data_temp[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        # y_batch, _embedding_yp_img = sess.run([tf.nn.l2_normalize(embedding, dim=1), embedding_yp_img],
        #                    feed_dict={x_raw: x_batch_data,
        #                               label_raw: c_batch_data,
        #                               is_Training: False})

        y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1)],
                                              feed_dict={x_raw: x_batch_data,
                                                         label_raw: c_batch_data,
                                                         is_Training: False})

        # Find predictions of classes that are not in the dataset.
        y_batch_data = y_batch[0]
        y_batches.append(y_batch_data)
        c_batches.append(c_batch_data)

        if visualized_emb:
            for x in x_batch_data:
                xx = cv2.cvtColor(cv2.resize(x + image_mean[0], (32, 32)), cv2.COLOR_BGR2GRAY)  # <class 'tuple'>: (3, 32, 32)
                test_img.append(xx / 255.0)

    y_data = np.concatenate(y_batches)
    c_data = np.concatenate(c_batches)

    # # 可视化重建图像
    # plt.imshow(_embedding_yp_img[0])

    # 训练集
    y_batches_train = []
    c_batches_train = []
    # for batch in tqdm(copy.copy(stream_train.get_epoch_iterator())):
    for batch in copy.copy(stream_train.get_epoch_iterator()):
        x_batch_data_temp, c_batch_data = batch  #
        x_batch_data = np.transpose(x_batch_data_temp[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1)],
                           feed_dict={x_raw: x_batch_data,
                                      label_raw: c_batch_data,
                                      is_Training: False})
        # Find predictions of classes that are not in the dataset.
        y_batch_data = y_batch[0]
        y_batches_train.append(y_batch_data)  # 向量
        c_batches_train.append(c_batch_data)
        if visualized_emb:
            for x in x_batch_data:
                # xx = cv2.resize(x, (32, 32)).transpose(2, 0, 1)
                # trian_img.append((xx + image_mean[0].transpose(2, 0, 1)) / 255.0)
                xx = cv2.cvtColor(cv2.resize(x + image_mean[0], (32, 32)),
                                  cv2.COLOR_BGR2GRAY)  # <class 'tuple'>: (3, 32, 32)
                trian_img.append(xx / 255.0)

    y_data_t = np.concatenate(y_batches_train)
    c_data_t = np.concatenate(c_batches_train)
    if visualized_tsne:
        labels = np.concatenate(c_data_t)
        labels_name = []
        for i in labels:
            if i == 0:
                labels_name.append('benign')
            else:
                labels_name.append('malignance')
        save_path = os.path.join(model_path, 'tsne')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot_tsne(y_data_t, labels_name, os.path.join(save_path, 'trainSet.png'))  # 训练集

        labels = np.concatenate(c_data)
        labels_name = []
        # 保证第一个元素是恶性
        if not labels[0] == 1:
            for i in labels:
                if i==1:
                    break
            labels[0] = 1
            labels[i] = 0
            temp_emb = y_data[0]
            y_data[0] = y_data[i]
            y_data[i] = temp_emb

        for i in labels:
            if i == 0:
                labels_name.append('benign')
            else:
                labels_name.append('malignance')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plot_tsne(y_data, labels_name, os.path.join(save_path, 'testSet.png'))  # 测试集

    if visualized_emb:
        # # 生成Sprite图像
        sprite_image = create_sprite_image(trian_img)
        # 放到日志目录下
        path_for_prostate_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
        plt.imsave(path_for_prostate_sprites, sprite_image, cmap='gray')
        plt.imshow(sprite_image, cmap='gray')
        # 标签文件写入
        path_for_prostate_metadata = os.path.join(LOG_DIR, META_FILE)
        path_for_prostate_tensor = os.path.join(LOG_DIR, TENSOR_FILE)
        with open(path_for_prostate_metadata, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(c_data_t):
                f.write("%d\t%d\n" % (index, label))
        with open(path_for_prostate_tensor, 'w') as f:
            for index, embedding in enumerate(y_data_t):
                f.write('\t'.join([str(x) for x in embedding]) + "\n")

        # 生成可视化向量所需要的日志问价
        # embedding_Visual(y_data_t, 'trainset_embedding', LOG_DIR, META_FILE, SPRITE_FILE, TRAINING_STEPS=8000)
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/experiment')
        # writer.add_embedding(y_data_t, c_data_t,
        #                      torch.from_numpy(np.stack(test_img, axis=0)),
        #                      global_step=8000, tag='embedding/train')
        # writer.flush()

    n_clusters = num_class
    nmi, f1 = evaluate_cluster(y_data, c_data, n_clusters)

    recalls_zsl, recall, precision, f1_score, max_f1score = evaluate_recall_according_trainset(features_test=y_data, features_train=y_data_t, labels_train=c_data_t,
                                  labels_test=c_data, neighbours=neighb)

    print(nmi)
    print(f1)
    return nmi, f1, recalls_zsl, recall, precision, f1_score, max_f1score

def Evaluation_with_crossent(stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding,
               predict_label, origin_label):
    """

    :param stream_test: 测试集数据流
    :param image_mean:
    :param sess:
    :param x_raw: 占位符
    :param label_raw:  占位符
    :param is_Training:占位符
    :param embedding:
    :param num_class:
    :param neighb:
    :return:
    """
    y_batches = []
    c_batches = []
    predict_by_ents = []
    true_labels = []
    for batch in copy.copy(stream_test.get_epoch_iterator()):
        x_batch_data, c_batch_data = batch  #
        x_batch_data = np.transpose(x_batch_data[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        y_batch, predict_by_ent, true_label = sess.run([tf.nn.l2_normalize(embedding, dim=1), predict_label, origin_label],
                           feed_dict={x_raw: x_batch_data,
                                      label_raw: c_batch_data,
                                      is_Training: False})
        # Find predictions of classes that are not in the dataset.
        y_batch_data = y_batch
        c_batch_data = np.reshape(c_batch_data, (-1))
        y_batches.append(y_batch_data)
        c_batches.append(c_batch_data)
        predict_by_ents.extend(predict_by_ent)
        true_labels.extend(true_label)
    y_data = np.concatenate(y_batches)
    c_data = np.concatenate(c_batches)

    metric = classification_report(true_labels, predict_by_ents, labels=None, target_names=None, sample_weight=None,
                                   digits=3)
    print(metric)

    # # 训练集
    # y_batches_train = []
    # c_batches_train = []
    # # for batch in tqdm(copy.copy(stream_train.get_epoch_iterator())):
    # for batch in copy.copy(stream_train.get_epoch_iterator()):
    #     x_batch_data, c_batch_data = batch  #
    #     x_batch_data = np.transpose(x_batch_data[:, [2, 1, 0], :, :], (0, 2, 3, 1))
    #     x_batch_data = x_batch_data - image_mean
    #     y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1)],
    #                        feed_dict={x_raw: x_batch_data,
    #                                   label_raw: c_batch_data,
    #                                   is_Training: False})
    #     # Find predictions of classes that are not in the dataset.
    #     y_batch_data = y_batch[0]
    #     y_batches_train.append(y_batch_data)
    #     c_batches_train.append(c_batch_data)
    # y_data_t = np.concatenate(y_batches_train)
    # c_data_t = np.concatenate(c_batches_train)
    #
    #
    # n_clusters = num_class
    nmi, f1 = evaluate_cluster(y_data, c_data, n_clusters=2)
    #
    # recalls_zsl, recall, precision, f1_score, max_f1score = evaluate_recall_according_trainset(features_test=y_data, features_train=y_data_t, labels_train=c_data_t,
    #                               labels_test=c_data, neighbours=neighb)
    #
    # # print(nmi)
    # # print(f1)
    # nmi = 0
    # f1= 0
    recalls_zsl = 0
    # f1_score = 0
    recall = recall_score(true_labels, predict_by_ents, average='weighted')
    f1score = f1_score(true_labels, predict_by_ents, average='weighted')
    max_f1score = f1score
    precision = precision_score(true_labels, predict_by_ents, average='weighted')
    return nmi, f1, recalls_zsl, recall, precision, f1score, max_f1score


# def Evaluation(stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding, num_class, neighb):
#     """
#
#     :param stream_test: 测试集数据流
#     :param image_mean:
#     :param sess:
#     :param x_raw: 占位符
#     :param label_raw:  占位符
#     :param is_Training:
#     :param embedding:
#     :param num_class:
#     :param neighb:
#     :return:
#     """
#     y_batches = []
#     c_batches = []
#     y_pred = []
#     y_true = []
#     for batch in tqdm(copy.copy(stream_test.get_epoch_iterator())):
#         x_batch_data, c_batch_data = batch  #
#         x_batch_data = np.transpose(x_batch_data[:, [2, 1, 0], :, :], (0, 2, 3, 1))
#         x_batch_data = x_batch_data - image_mean
#         y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1)],
#                            feed_dict={x_raw: x_batch_data,
#                                       label_raw: c_batch_data,
#                                       is_Training: False})
#         # Find predictions of classes that are not in the dataset.
#         # y_pred.extend(list(np.argmax(y_batch[1], axis=1)))
#         # y_true_batch = y_batch[2]
#         # y_true.extend(list(y_true_batch))
#         y_batch_data = y_batch[0]
#         # y_pred = y_batch[1]
#         y_batches.append(y_batch_data)
#         c_batches.append(c_batch_data)
#
#     y_data = np.concatenate(y_batches)
#     c_data = np.concatenate(c_batches)
#     n_clusters = num_class
#     nmi, f1 = evaluate_cluster(y_data, c_data, n_clusters)
#
#     recalls = evaluate_recall(y_data, c_data, neighbours=neighb)
#
#     # metric = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=3)
#     # print(metric)
#     print(nmi)
#     print(f1)
#     return nmi, f1, recalls


# def Evaluation(stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding, num_class, neighb, pred_class_ids, label):
#     """
#
#     :param stream_test: 测试集数据流
#     :param image_mean:
#     :param sess:
#     :param x_raw: 占位符
#     :param label_raw:  占位符
#     :param is_Training:
#     :param embedding:
#     :param num_class:
#     :param neighb:
#     :return:
#     """
#     y_batches = []
#     c_batches = []
#     y_pred = []
#     y_true = []
#     for batch in tqdm(copy.copy(stream_test.get_epoch_iterator())):
#         x_batch_data, c_batch_data = batch  #
#         x_batch_data = np.transpose(x_batch_data[:, [2,1,0], :, :], (0, 2, 3, 1))
#         x_batch_data = x_batch_data-image_mean
#         y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1), pred_class_ids, label],
#                            feed_dict={x_raw: x_batch_data,
#                                       label_raw: c_batch_data,
#                                       is_Training: False})
#         # Find predictions of classes that are not in the dataset.
#         y_pred.extend(list(np.argmax(y_batch[1], axis=1)))
#         y_true_batch = y_batch[2]
#         y_true.extend(list(y_true_batch))
#         y_batch_data = y_batch[0]
#         # y_pred = y_batch[1]
#         y_batches.append(y_batch_data)
#         c_batches.append(c_batch_data)
#
#     y_data = np.concatenate(y_batches)
#     c_data = np.concatenate(c_batches)
#     n_clusters = num_class
#     nmi, f1 = evaluate_cluster(y_data, c_data, n_clusters)
#     recalls = evaluate_recall(y_data, c_data, neighbours=neighb)
#     metric = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=3)
#     print(metric)
#     print(nmi)
#     print(f1)
#     return nmi, f1, recalls


def products_Evaluation(stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding, num_class, neighb):
    y_batches = []
    c_batches = []
    for batch in tqdm(copy.copy(stream_test.get_epoch_iterator())):
        x_batch_data, c_batch_data = batch
        x_batch_data = np.transpose(x_batch_data[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        y_batch = sess.run([tf.nn.l2_normalize(embedding, dim=1)],
                           feed_dict={x_raw: x_batch_data, label_raw: c_batch_data, is_Training: False})
        y_batch_data = y_batch[0]
        y_batches.append(y_batch_data)
        c_batches.append(c_batch_data)
    y_data = np.concatenate(y_batches)
    c_data = np.concatenate(c_batches)
    recalls = evaluate_recall(y_data, c_data, neighbours=neighb)
    return recalls


def Embedding_Saver(stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding, savepath, step):
    y_batches = []
    c_batches = []
    for batch in tqdm(copy.copy(stream_test.get_epoch_iterator())):
        x_batch_data, c_batch_data = batch
        x_batch_data = np.transpose(x_batch_data[:, [2, 1, 0], :, :], (0, 2, 3, 1))
        x_batch_data = x_batch_data - image_mean
        y_batch = sess.run([embedding], feed_dict={x_raw: x_batch_data, label_raw: c_batch_data, is_Training: False})
        y_batch_data = y_batch[0]
        y_batches.append(y_batch_data)
        c_batches.append(c_batch_data)
    y_data = np.concatenate(y_batches)
    c_data = np.concatenate(c_batches)
    np.save(savepath + str(step) + '-y_batch.npy', y_data)
    np.save(savepath + str(step) + '-c_batch.npy', c_data)
    print('embedding saved')


def Embedding_Evaler(step, path, num_class, is_nmi, is_recall, neighb):
    path = path + step + '-'
    y_data = np.load(path + 'y_batch.npy')
    c_data = np.load(path + 'c_batch.npy')
    print("starts")
    print(step)
    if is_nmi:
        nmi, f1 = evaluate_cluster(y_data, c_data, num_class)
        print('nmi: %f' % nmi)
        print('f1: %f' % f1)
    if is_recall:
        recalls = evaluate_recall(y_data, c_data, neighb)
        for i in range(0, np.shape(recalls)[0]):
            print('Recall@%d: %f' % (neighb[i], recalls[i]))
