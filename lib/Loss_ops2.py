from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import lib.nn_Ops as nn_Ops
from flags.FLAGS_HDML_triplet import *
"""
Classic Metric Learning Losses
"""


def remove_zero(has_zero):
    return tf.reshape(tf.gather(params=has_zero, indices=tf.where(tf.not_equal(has_zero, tf.zeros(shape=tf.shape(has_zero), dtype=tf.float32)))), [-1])


def all_in(loss_vector):
    mean, var = tf.nn.moments(loss_vector, axes=0, keep_dims=False)
    return tf.logical_and(tf.not_equal(tf.shape(tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
                                    tf.greater(loss_vector, (mean+3.*tf.sqrt(var))*tf.ones(tf.shape(loss_vector), dtype=tf.float32))
                                )), [-1]))[0], 0), tf.not_equal(tf.shape(tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
                                    tf.greater((mean-3.*tf.sqrt(var))*tf.ones(tf.shape(loss_vector), dtype=tf.float32), loss_vector)
                                )), [-1]))[0], 0))


def remove(loss_vector):
    loss_vector = tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
        tf.greater(tf.reduce_max(loss_vector) * tf.ones(tf.shape(loss_vector), dtype=tf.float32), loss_vector)
    )), [-1])
    loss_vector = tf.reshape(tf.gather(params=loss_vector, indices=tf.where(
        tf.greater(loss_vector, tf.reduce_min(loss_vector) * tf.ones(tf.shape(loss_vector), dtype=tf.float32))
    )), [-1])
    return loss_vector


def trunc_loss(loss_vector):
    loss_vector = tf.reshape(loss_vector, shape=[tf.shape(loss_vector)[0]])

    loss_vector = tf.while_loop(all_in, remove, [loss_vector])
    return loss_vector


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
        feature: 2-D Tensor of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.
    Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(
            math_ops.square(feature),
            axis=[1],
            keep_dims=True),
        math_ops.reduce_sum(
            math_ops.square(
                array_ops.transpose(feature)),
            axis=[0],
            keep_dims=True)) - 2.0 * math_ops.matmul(
        feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
    """Computes the contrastive loss.
    This loss encourages the embedding to be close to each other for
        the samples of the same label and the embedding to be far apart at least
        by the margin constant for the samples of different labels.
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
            binary labels indicating positive vs negative pair.
        embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
            images. Embeddings should be l2 normalized.
        embeddings_positive: 2-D float `Tensor` of embedding vectors for the
            positive images. Embeddings should be l2 normalized.
        margin: margin term in the loss definition.
    Returns:
        contrastive_loss: tf.float32 scalar.
    """
    # Get per pair distances
    distances = math_ops.sqrt(
        math_ops.reduce_sum(
            math_ops.square(embeddings_anchor - embeddings_positive), 1))

    # Add contrastive loss for the siamese network.
    #   label here is {0,1} for neg, pos.  减小同类距离，增大类间距离
    return math_ops.reduce_mean(
        math_ops.to_float(labels) * math_ops.square(distances) +
        (1. - math_ops.to_float(labels)) *
        math_ops.square(math_ops.maximum(margin - distances, 0.)),
        name='contrastive_loss')


def contrastive_loss_v2(pos1, pos2, neg1, neg2, alpha=1.0):  # 根据分配方案可知， pos1与pos2同类， neg1与neg2异类
    distance = tf.reduce_sum((pos1 - pos2) ** 2.0, axis=1) + \
               tf.nn.relu(
                   alpha - tf.reduce_sum((neg1 - neg2) ** 2.0, axis=1))
    return tf.reduce_mean(distance) / 4


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
      Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the maximum.
      Returns:
        masked_maximums: N-D `Tensor`.
          The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(
            data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
    axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(
            data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums
    return masked_minimums


def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(d_ap - d_an + margin, 0.)
    return tf.reduce_sum(loss)


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
      The loss encourages the positive distances (between a pair of embeddings with
      the same labels) to be smaller than the minimum negative distance among
      which are at least greater than the positive distance plus the margin constant
      (called semi-hard negative) in the mini-batch. If no such negative exists,
      uses the largest negative distance instead.
      See: https://arxiv.org/abs/1503.03832.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        triplet_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(
                    mask, dtype=dtypes.float32), 1, keep_dims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    _triplet_loss = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    return _triplet_loss


def new_npair_loss(labels, embedding_anchor, embedding_positive, reg_lambda, equal_shape=True, half_batch_size=FLAGS.batch_size//2):
    """
    论文：For the N-pair loss [28], we also use the distance of the
          positive pair as the reference distance, but generate all the
          N − 1 negatives for each anchor in an (N+1)-tuple:
    N-pair loss used in HDML, the inputs are constructed into N/2 N/2+1 pairs
    :param labels: A 1-d tensor of size [batch_size], which presents the sparse label of the embedding
            <class 'list'>: [array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int32), array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int32)]
    :param embedding_anchor: A 4-d tensor of size [batch_size/2, H, W, C], the embedding of anchor

    :param embedding_positive: A 4-d tensor of size [batch_size/2, H, W, C], the embedding of positive
    :param reg_lambda: float, the l2-regular factor of N-pair Loss
    :param equal_shape: boolean, whether shape(embedding_anchor)[0] == shape(embedding_positive)[0]
    :param half_batch_size: int, if batch size == 128, half_batch_size will be 64
    :return: The n-pair loss, which equals to npair_loss + reg_lambda*l2_loss
    """
    reg_anchor = math_ops.reduce_mean(  # 356.57693  所有anchor求平均
        math_ops.reduce_sum(math_ops.square(embedding_anchor), 1))
    reg_positive = math_ops.reduce_mean(  # 224.5543  # 所有positive求平均
        math_ops.reduce_sum(math_ops.square(embedding_positive), 1))
    l2loss = math_ops.multiply(   # l2正则化损失
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')
    xent_loss = []
    if equal_shape:  # 如果锚与正例数相等则 展开为（8*8,128）的形状
        pos_tile = tf.tile(embedding_positive, [half_batch_size, 1], name='pos_tile')
    else:
        pos_tile = embedding_positive  # （64， 128）
    # 样本
    anc = tf.split(embedding_anchor, half_batch_size, axis=0)  # 8个（1，128）的tensor
    pos = tf.split(pos_tile, half_batch_size, axis=0)  # 8个（8，128）的tensor,每一个看起来都是一样的
    # 标签
    label2 = tf.split(labels, 2, axis=0)  # 2个（8，）的tensor
    label_anc = tf.reshape(label2[0], [half_batch_size, 1])   # (8,1)
    label_pos = tf.reshape(label2[1], [half_batch_size, 1])  # (8,1)
    label_anc = tf.split(label_anc, half_batch_size, axis=0)  # 8个（1，1）的tensor

    # 循环计算每一个anchor与所有pos的相似度矩阵，并求交叉熵损失
    for i in range(half_batch_size):
        # （1，128）* （8，128）T--->（1，8）   # (1,8)  每一个anchor与8个pos的点积总和，和越大，相似度越大

        similarity_matrix = tf.matmul(anc[i], pos[i], transpose_a=False, transpose_b=True)

        anc_label = tf.reshape(label_anc[i], [1, 1])  # [0]-->[[0]]
        pos_label = tf.reshape(label_pos, [half_batch_size, 1])  # （8,1）
        # 当前anchor与所有pos的标签是否一致
        labels_remapped = tf.to_float(  # [[0. 0. 0. 0. 1. 1. 1. 1.]]
            tf.equal(anc_label, tf.transpose(pos_label))
        )
        # [[0.   0.   0.   0.   0.25 0.25 0.25 0.25]]
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keep_dims=True)

        x_loss = tf.nn.softmax_cross_entropy_with_logits(  # [31.79718]
            logits=similarity_matrix, labels=labels_remapped
        )
        xent_loss.append(x_loss)

    xent_loss = tf.reduce_mean(xent_loss, name='xentrop')
    # 为空则取0，否则为xent_loss + l2loss
    r_loss = tf.cond(tf.is_nan(xent_loss + l2loss), lambda: tf.constant(0.), lambda: xent_loss + l2loss)
    return r_loss, reg_anchor, reg_positive, l2loss, pos_tile, anc, pos, label2, label_anc, label_pos, label_anc, similarity_matrix, labels_remapped, x_loss


def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=3e-3, print_losses=False):
    """Computes the npairs loss.
          Npairs loss expects paired data where a pair is composed of samples from the
          same labels and each pairs in the minibatch have different labels. The loss
          has two components. The first component is the L2 regularizer on the
          embedding vectors. The second component is the sum of cross entropy loss
          which takes each row of the pair-wise similarity matrix as logits and
          the remapped one-hot labels as labels.
          See:
          http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
          Args:
            labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
            embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
              embedding vectors for the anchor images. Embeddings should not be
              l2 normalized.
            embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
              embedding vectors for the positive images. Embeddings should not be
              l2 normalized.
            reg_lambda: Float. L2 regularization term on the embedding vectors.
            print_losses: Boolean. Option to print the xent and l2loss.
          Returns:
            npairs_loss: tf.float32 scalar.
      """
    # pylint: enable=line-too-long
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    # 这儿计算anchors和positives.T的内积，度量两两向量之间的距离
    # 每一行表示一个anchor与每个positives矩阵的元素内积的结果
    similarity_matrix = math_ops.matmul(
        embeddings_anchor, embeddings_positive, transpose_a=False,
        transpose_b=True)

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # 为了确定哪些距离需要拉近，哪些需要拉远，需要一个labels矩阵，
    # labels_remapped值为1的地方表示similarity_matrix对应地方的距离是需要拉近的（他们来自同一类），
    # 值为0的地方对应的距离需要拉远（不是同一类）
    labels_remapped = math_ops.to_float(
        math_ops.equal(labels, array_ops.transpose(labels)))
    # 归一化同一行label
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    # 论文中也提到用交叉熵的方式计算损失
    xent_loss = nn.softmax_cross_entropy_with_logits(
        logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(
            xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
    # r_loss = tf.cond(tf.is_nan(xent_loss + l2loss), lambda: tf.constant(0.), lambda: xent_loss + l2loss)  # 加
    return xent_loss + l2loss


def Mat(labels):
    N = tf.shape(labels)[0]   # [16]
    # 重复16遍 [16,16]
    # Mask = labels.repeat(N, 1)
    # Mask = tf.tile(tf.expand_dims(labels, axis=0), multiples=[N,1])
    # Reshape label tensor to [batch, 1]
    lshape = tf.shape(labels)
    # labels = tf.Print(labels, [labels], message='标签:', summarize=1000)
    labels = tf.reshape(labels, [lshape[0], 1])

    # build pairswise binary adjacency matrix
    same = tf.math.equal(labels, tf.transpose(labels))
    # same = tf.Print(same, [same], message='是否相同标签:', summarize=1000)

    # # Mask 和Mask的转置 对应元素是否同类， 相当于16个元素两两配对，对角线元素应该是元素与它本身配对
    # Same = math_ops.equal(Mask, tf.transpose(Mask))
    # # 对角线元素补充为0，即设为false
    same = tf.matrix_set_diag(same, tf.cast(tf.zeros([N]), tf.bool))
    diff = tf.logical_not(same)

    return same, diff


def distMC(Mat_A, Mat_B, norm=1, sq=True):
    N_A = tf.shape(Mat_A)[0]
    # Mat_A = tf.Print(Mat_A, [Mat_A], message='Mat_A:', summarize=1000)
    DC = math_ops.matmul(
        Mat_A, Mat_B, transpose_a=False,
        transpose_b=True)
    # DC = tf.Print(DC, [DC], message='未处理对角:', summarize=1000)
    if sq:
        DC = tf.matrix_set_diag(DC, tf.multiply(tf.cast(-norm, tf.float32), tf.ones([N_A])))

        # DC = tf.Print(DC, [DC], message='已处理对角:', summarize=1000)
    # DC = tf.Print(DC, [DC], "DC", summarize=1000)
    return DC

def dist_calculate(V_A, V_B, norm=1):
    """
    只取对角线元素， 即A_B对应元素内积
    :param V_A:  [18, 128]
    :param V_B:  [18, 128]

    """
    DC = math_ops.matmul(
        V_A, V_B, transpose_a=False,
        transpose_b=True)
    # DC = tf.Print(DC, [DC], message='未处理对角:', summarize=1000)
    # 取对角线元素
    DC = tf.diag_part(DC)
    return DC

# def easy_positive_hard_negative_loss(embeddings, labels,
#                 reg_lambda=3e-3, print_losses=False):
#     """Computes the npairs loss.
#           Npairs loss expects paired data where a pair is composed of samples from the
#           same labels and each pairs in the minibatch have different labels. The loss
#           has two components. The first component is the L2 regularizer on the
#           embedding vectors. The second component is the sum of cross entropy loss
#           which takes each row of the pair-wise similarity matrix as logits and
#           the remapped one-hot labels as labels.
#           See:
#           http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#           Args:
#             labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
#             embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
#               embedding vectors for the anchor images. Embeddings should not be
#               l2 normalized.
#             embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
#               embedding vectors for the positive images. Embeddings should not be
#               l2 normalized.
#             reg_lambda: Float. L2 regularization term on the embedding vectors.
#             print_losses: Boolean. Option to print the xent and l2loss.
#           Returns:
#             npairs_loss: tf.float32 scalar.
#       """
#     # pylint: enable=line-too-long
#     # Add the regularizer on the embedding.
#
#     # 正则化与其损失
#     reg_embeddings = math_ops.reduce_mean(
#         math_ops.reduce_sum(math_ops.square(embeddings), 1))
#     l2loss = math_ops.multiply(
#         0.25 * reg_lambda, reg_embeddings, name='l2loss')
#
#     same, Diff = Mat(labels)
#
#     # Get per pair similarities.
#     Dist = distMC(embeddings, embeddings)
#
#
#     ############################################
#     # finding max similarity on same label pairs寻找相同标签 相似度最大的pairs
#     #############################################
#     D_Positive = Dist  #  tf.stop_gradient??
#     # 把异类对的距离设为最大
#     # D_Positive[Diff] = -1
#     dis_mask = tf.ones_like(D_Positive)
#     D_Positive = tf.where(Diff, 0-dis_mask, D_Positive)
#     # D_Positive = tf.multiply(tf.cast(same, tf.float32), D_Positive) - tf.cast(Diff, tf.float32)
#
#     # D_Positive[D_Positive > 0.9999] = -1
#     D_Positive = tf.where(tf.greater(D_Positive, 0.9999), 0 - dis_mask, D_Positive)
#     # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号
#     V_pos  = tf.reduce_max(D_Positive, 1)
#     I_pos = tf.arg_max(D_Positive, 1)
#     Mask_not_drop_pos = (V_pos > 0)
#
#     # extracting pos score  提取位置上的相似度
#     Pos_log = V_pos
#
#     ############################################
#     # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
#     #############################################
#     D_Negative = Dist
#     # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
#     D_Negative = tf.where(same, 0 - dis_mask, D_Negative)
#     if FLAGS.semi:  # 是否采取semi hard negative采样
#         # choose a anchor-negative pair that is father than
#         # the anchor-positive pair, but within a margin
#         N = tf.shape(embeddings)[0]
#         # pp = V_pos.repeat(N, 1).t()
#
#         mask_semi = (D_Negative > tf.tile(tf.expand_dims(V_pos, axis=0), multiples=[N, 1])) & Diff
#         D_Negative = tf.where(mask_semi, 0 - dis_mask, D_Negative)
#         # D_detach_N[(D_detach_N > (pp)) & Diff] = -1  # extracting SHN
#     V_neg = tf.reduce_max(D_Negative, 1)
#     I_neg = tf.arg_max(D_Negative, 1)
#
#     # prevent duplicated pairs  # 排除样本与样本本身组成的pair
#     Mask_not_drop_neg = (V_neg > 0)
#
#     # extracting neg score  提取位置上的相似度
#     Neg_log = V_neg
#
#     # triplets  ###############根据提取的正对负对构建三元组#########################
#     trpl = tf.stack([V_pos, V_neg], 1)
#     Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg
#
#     # loss triplet_L=max(0, dap-dan+margin)
#     # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
#     Prob = tf.nn.log_softmax(trpl / 0.1, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
#     # loss = Prob[Mask_not_drop].mean()
#     pro_mask = tf.zeros_like(Prob)
#     loss = tf.reduce_mean(tf.where(Mask_not_drop, Prob, pro_mask))
#
#     # # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
#     # lshape = array_ops.shape(labels)
#     # assert lshape.shape == 1
#     # labels = array_ops.reshape(labels, [lshape[0], 1])
#     #
#     # labels_remapped = math_ops.to_float(
#     #     math_ops.equal(labels, array_ops.transpose(labels)))
#     # labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)
#     #
#     # # Add the softmax loss.
#     # xent_loss = nn.softmax_cross_entropy_with_logits(
#     #     logits=similarity_matrix, labels=labels_remapped)
#     # xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')
#
#     # if print_losses:
#     #     xent_loss = logging_ops.Print(
#     #         xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
#     r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加
#     return r_loss


def _pairwise_distances(embeddings, squared=False):
    """
    计算嵌入向量之间的距离
    Args:
        embeddings: 形如(batch_size, embed_dim)的张量
        squared: Boolean. True->欧式距离的平方，False->欧氏距离
    Returns:
        piarwise_distances: 形如(batch_size, batch_size)的张量
    """
    # 嵌入向量点乘，输出shape=(batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # 取dot_product对角线上的值，相当于是每个嵌入向量的L2正则化，shape=(batch_size,)
    square_norm = tf.diag_part(dot_product)

    # 计算距离,shape=(batch_size, batch_size)
    # ||a - b||^2 = ||a||^2 - 2<a, b> + ||b||^2
    # PS: 下面代码计算的是||a - b||^2，结果是一样的
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # 保证距离都>=0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # 加一个接近0的值，防止求导出现梯度爆炸的情况
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # 校正距离
        distances = distances * (1.0 - mask)
    return distances



def easy_positive_hard_negative_loss1(embeddings, labels, temperate,
                                     reg_lambda=3e-3, print_losses=False):
    """

    :param embeddings:  l2norm之后的
    :param labels:
    :param temperate:
    :param reg_lambda:
    :param print_losses:
    :return:
    """
    # 正则化与其损失
    # reg_embeddings = math_ops.reduce_mean(
    #     math_ops.reduce_sum(math_ops.square(embeddings), 1))
    # l2loss = math_ops.multiply(
    #     0.25 * reg_lambda, reg_embeddings, name='l2loss')

    same, Diff = Mat(labels)

    # embeddings = tf.Print(embeddings, [embeddings], message='归一化前：', summarize=1000)

    # l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)

    # l2norm_emb = tf.Print(l2norm_emb, [l2norm_emb], message='l2归一化后：', summarize=1000)
    # Get per pair similarities.
    Dist = distMC(embeddings, embeddings)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)


    ############################################
    # finding max similarity on same label pairs寻找相同标签 相似度最大的pairs
    #############################################
    D_Positive = Dist  #  tf.stop_gradient??
    # 把异类对的距离设为最大
    # D_Positive[Diff] = -1
    dis_mask = tf.ones_like(D_Positive)
    D_Positive = tf.where(Diff, 0-dis_mask, D_Positive)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)
    # D_Positive = tf.multiply(tf.cast(same, tf.float32), D_Positive) - tf.cast(Diff, tf.float32)

    # D_Positive[D_Positive > 0.9999] = -1
    D_Positive = tf.where(tf.greater(D_Positive, 0.9999), 0 - dis_mask, D_Positive)
    # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号
    V_pos  = tf.reduce_max(D_Positive, axis=1)
    I_pos = tf.arg_max(D_Positive, 1)

    Mask_not_drop_pos = (V_pos > 0)

    # extracting pos score  提取位置上的相似度
    Pos_log = V_pos

    ############################################
    # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
    #############################################
    D_Negative = Dist
    # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
    D_Negative = tf.where(same, 0 - dis_mask, D_Negative)
    if FLAGS.semi:  # 是否采取semi hard negative采样
        # choose a anchor-negative pair that is father than
        # the anchor-positive pair, but within a margin
        N = tf.shape(embeddings)[0]
        # pp = V_pos.repeat(N, 1).t()
        # V_pos = tf.Print(V_pos, [V_pos], message='V_pos相似度', summarize=1000)
        pp = tf.transpose(tf.tile(tf.expand_dims(V_pos, axis=0), multiples=[N, 1]))
        # pp = tf.Print(pp, [pp], message='掩码', summarize=1000)
        mask_semi = (D_Negative > pp) & Diff
        D_Negative = tf.where(mask_semi, 0 - dis_mask, D_Negative)
        # D_detach_N[(D_detach_N > (pp)) & Diff] = -1  # extracting SHN
    # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    V_neg = tf.reduce_max(D_Negative, 1)
    I_neg = tf.arg_max(D_Negative, 1)
    # I_neg [9 10 14 13 15 10 13 16 15  0  1  4  0  3  2  4  5  3]
    # I_pos [ 3  6  6  6  8  7  3  5  7 17 17 16 14 17 12 12 11  9]

    # V_pos = tf.Print(V_pos, [V_pos], message='正对相似度', summarize=1000)
    # V_neg = tf.Print(V_neg, [V_neg], message='负对相似度', summarize=1000)
    # I_neg = tf.Print(I_neg, [I_neg], message='负对索引', summarize=1000)
    # I_pos = tf.Print(I_pos, [I_pos], message='正对索引', summarize=1000)

    # prevent duplicated pairs  # 排除样本与样本本身组成的pair

    Mask_not_drop_neg = (V_neg > 0)
    #
    # # extracting neg score  提取位置上的相似度
    # Neg_log = V_neg

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl = tf.stack([V_pos, V_neg], 1)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
    Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg
    # Mask_not_drop = tf.Print(Mask_not_drop, [Mask_not_drop], message='不丢弃：', summarize=1000)

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # logits = trpl / 0.11
    # margin = 1/(5+tf.exp(Jm_loss))
    # Jm_loss = tf.clip_by_value(Jm_loss, 0, 0.8)
    # margin = -Jm_loss/4+0.3
    # margin = tf.Print(margin, [margin], message='间距margin', summarize=1000)
    logits = trpl / temperate
    # logits = tf.Print(logits, [logits], message='损失logits', summarize=1000)
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # Prob = tf.Print(Prob, [Prob], message='log_softmax损失', summarize=1000)
    pro_mask = tf.zeros_like(Prob)
    trip_less_zero = tf.where(Mask_not_drop, Prob, pro_mask)
    # trip_less_zero = tf.Print(trip_less_zero, [trip_less_zero], message='剔除后的损失', summarize=1000)
    loss = tf.reduce_mean(trip_less_zero)
    # loss = tf.Print(loss, [loss], message='平均损失', summarize=1000)

    # if print_losses:
    #     xent_loss = logging_ops.Print(
    #         xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
    # r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加

    # r_loss = tf.cond(tf.is_nan(loss), lambda: tf.constant(0.), lambda: loss)  # 加
    positive_emb = tf.gather(embeddings, I_pos)
    negative_emb = tf.gather(embeddings, I_neg)
    positive_label = tf.gather(labels, I_pos)
    negative_label = tf.gather(labels, I_neg)
    return loss, I_pos, positive_emb, I_neg, negative_emb, positive_label, negative_label


def easy_positive_hard_negative_loss(anchor, positive, negative):
    # Get per pair similarities.
    # 这儿计算anchors和positives.T的内积，度量两两向量之间的距离
    # 每一行表示一个anchor与每个positives矩阵的元素内积的结果
    similarity_matrix_ap = math_ops.matmul(
        anchor, positive, transpose_a=False,
        transpose_b=True)
    similarity_matrix_an = math_ops.matmul(
        anchor, negative, transpose_a=False,
        transpose_b=True)
    easy_positive = tf.reduce_max(similarity_matrix_ap, axis=1)
    hard_negative = tf.reduce_max(similarity_matrix_an, axis=1)
    # loss = tf.reduce_mean(-tf.log(tf.exp(easy_positive)/(easy_positive + hard_negative)))
    trpl = tf.stack([easy_positive, hard_negative], 1)
    logits = trpl / 0.1
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]
    loss = tf.reduce_mean(Prob)
    return loss


def label_preserving_loss(l2norm_emb, labels, temperate,
                                     reg_lambda=3e-3, print_losses=False):
    # 正则化与其损失
    # reg_embeddings = math_ops.reduce_mean(
    #     math_ops.reduce_sum(math_ops.square(embeddings), 1))
    # l2loss = math_ops.multiply(
    #     0.25 * reg_lambda, reg_embeddings, name='l2loss')

    same, Diff = Mat(labels)

    # Get per pair similarities.
    Dist = distMC(l2norm_emb, l2norm_emb)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)


    ############################################
    # finding max similarity on same label pairs寻找相同标签 相似度最大的pairs
    #############################################
    D_Positive = Dist  #  tf.stop_gradient??
    # 把异类对的距离设为最大
    # D_Positive[Diff] = -1
    dis_mask = tf.ones_like(D_Positive)
    D_Positive = tf.where(Diff, 0-dis_mask, D_Positive)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)
    # D_Positive = tf.multiply(tf.cast(same, tf.float32), D_Positive) - tf.cast(Diff, tf.float32)

    # D_Positive[D_Positive > 0.9999] = -1
    D_Positive = tf.where(tf.greater(D_Positive, 0.9999), 0 - dis_mask, D_Positive)
    # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号

    # 找hardest positive support sample
    indx = tf.range(FLAGS.batch_size, tf.shape(labels)[0])

    # 随机取batchsize个作为anchor
    # indx_batchsize = tf.range(FLAGS.batch_size)
    # indx = tf.random_shuffle(tf.range(tf.shape(labels)[0]))
    # indx = tf.gather(indx, indx_batchsize, axis=0)


    dist_ap = tf.gather(D_Positive, indx, axis=0)
    dist_ap = tf.gather(dist_ap, indx, axis=1)

    V_pos = tf.reduce_min(dist_ap, axis=1)  # hardest positive
    I_pos = tf.arg_min(dist_ap, 1)

    # 合成样本也参与
    # dist_ap = tf.gather(D_Positive, indx, axis=0)  # 18,54
    # V_pos = tf.reduce_max(dist_ap, 1)

    Mask_not_drop_pos = (V_pos > 0)

    ############################################
    # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
    #############################################
    D_Negative = Dist
    # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
    D_Negative = tf.where(same, 0 - dis_mask, D_Negative)  # 54 x 54

    # 取最难点要包括合成点
    # num_group = tf.shape(labels)[0] // FLAGS.batch_size
    # D_Negative = tf.reduce_max(D_Negative, 1)  # 54
    # dist_an = tf.reshape(D_Negative, [num_group, FLAGS.batch_size])  # <class 'tuple'>: (3, 18)  Input to reshape is a tensor with 2916 values
    # # dist_an = F.min(temp, axis=0)  # include synthetic positives  包含合成的点
    # # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    # V_neg = tf.reduce_max(dist_an, 0)  # 54


    dist_an = tf.gather(D_Negative, indx, axis=0)  # 18,54
    # D_Negative = tf.reduce_max(D_Negative, 1)  # 18
    # dist_an = tf.reshape(D_Negative, [num_group, FLAGS.batch_size])
    V_neg = tf.reduce_max(dist_an, 1)

    # I_neg = tf.arg_max(D_Negative, 1)
    # I_neg [9 10 14 13 15 10 13 16 15  0  1  4  0  3  2  4  5  3]
    # I_pos [ 3  6  6  6  8  7  3  5  7 17 17 16 14 17 12 12 11  9]

    # V_pos = tf.Print(V_pos, [V_pos], message='正对相似度', summarize=1000)
    # V_neg = tf.Print(V_neg, [V_neg], message='负对相似度', summarize=1000)
    # I_neg = tf.Print(I_neg, [I_neg], message='负对索引', summarize=1000)
    # I_pos = tf.Print(I_pos, [I_pos], message='正对索引', summarize=1000)

    # prevent duplicated pairs  # 排除样本与样本本身组成的pair

    Mask_not_drop_neg = (V_neg > 0)
    #
    # # extracting neg score  提取位置上的相似度
    # Neg_log = V_neg

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl = tf.stack([V_pos, V_neg], 1)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
    Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg
    # Mask_not_drop = tf.Print(Mask_not_drop, [Mask_not_drop], message='不丢弃：', summarize=1000)

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # logits = trpl / 0.11
    # margin = 1/(5+tf.exp(Jm_loss))
    # Jm_loss = tf.clip_by_value(Jm_loss, 0, 0.8)
    # margin = -Jm_loss/4+0.3
    # margin = tf.Print(margin, [margin], message='间距margin', summarize=1000)


    logits = trpl / temperate
    # logits = tf.Print(logits, [logits], message='损失logits', summarize=1000)
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN


    # Prob = tf.Print(Prob, [Prob], message='log_softmax损失', summarize=1000)
    pro_mask = tf.zeros_like(Prob)
    trip_less_zero = tf.where(Mask_not_drop, Prob, pro_mask)
    # trip_less_zero = tf.Print(trip_less_zero, [trip_less_zero], message='剔除后的损失', summarize=1000)
    loss = tf.reduce_mean(trip_less_zero)  ##
    # loss = tf.Print(loss, [loss], message='平均损失', summarize=1000)

    # if print_losses:
    #     xent_loss = logging_ops.Print(
    #         xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
    # r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加

    # r_loss = tf.cond(tf.is_nan(loss), lambda: tf.constant(0.), lambda: loss)  # 加
    # positive_emb = tf.gather(embeddings, I_pos)
    # negative_emb = tf.gather(embeddings, I_neg)
    # positive_label = tf.gather(labels, I_pos)
    # negative_label = tf.gather(labels, I_neg)
    return loss

def ephn_label_preserving_loss(l2norm_emb, labels, temperate,
                                     reg_lambda=3e-3, print_losses=False):
    # 正则化与其损失
    # reg_embeddings = math_ops.reduce_mean(
    #     math_ops.reduce_sum(math_ops.square(embeddings), 1))
    # l2loss = math_ops.multiply(
    #     0.25 * reg_lambda, reg_embeddings, name='l2loss')

    same, Diff = Mat(labels)

    # Get per pair similarities.
    Dist = distMC(l2norm_emb, l2norm_emb)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)


    ############################################
    # finding max similarity on same label pairs寻找相同标签 相似度最大的pairs
    #############################################
    D_Positive = Dist  #  tf.stop_gradient??
    # 把异类对的距离设为最大
    # D_Positive[Diff] = -1
    dis_mask = tf.ones_like(D_Positive)
    D_Positive = tf.where(Diff, 0-dis_mask, D_Positive)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)
    # D_Positive = tf.multiply(tf.cast(same, tf.float32), D_Positive) - tf.cast(Diff, tf.float32)

    # D_Positive[D_Positive > 0.9999] = -1
    D_Positive = tf.where(tf.greater(D_Positive, 0.9999), 0 - dis_mask, D_Positive)
    # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号
    # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号

    # # anc包括原样本和合成样本， 但正例只在原样本里找
    # indx = tf.range(FLAGS.batch_size)
    # V_pos = tf.gather(D_Positive, indx, axis=0)

    V_pos  = tf.reduce_max(D_Positive, axis=1)
    I_pos = tf.arg_max(D_Positive, 1)
    Mask_not_drop_pos = (V_pos > 0)

    # extracting pos score  提取位置上的相似度
    Pos_log = V_pos

    ############################################
    # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
    #############################################
    D_Negative = Dist
    # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
    D_Negative = tf.where(same, 0 - dis_mask, D_Negative)

    # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    V_neg = tf.reduce_max(D_Negative, 1)
    I_neg = tf.arg_max(D_Negative, 1)
    Mask_not_drop_neg = (V_neg > 0)

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl = tf.stack([V_pos, V_neg], 1)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
    Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg
    # Mask_not_drop = tf.Print(Mask_not_drop, [Mask_not_drop], message='不丢弃：', summarize=1000)

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # logits = trpl / 0.11
    # margin = 1/(5+tf.exp(Jm_loss))
    # Jm_loss = tf.clip_by_value(Jm_loss, 0, 0.8)
    # margin = -Jm_loss/4+0.3
    # margin = tf.Print(margin, [margin], message='间距margin', summarize=1000)
    logits = trpl / temperate
    # logits = tf.Print(logits, [logits], message='损失logits', summarize=1000)
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # Prob = tf.Print(Prob, [Prob], message='log_softmax损失', summarize=1000)
    pro_mask = tf.zeros_like(Prob)
    trip_less_zero = tf.where(Mask_not_drop, Prob, pro_mask)
    # trip_less_zero = tf.Print(trip_less_zero, [trip_less_zero], message='剔除后的损失', summarize=1000)
    loss = tf.reduce_mean(trip_less_zero)


    # --------------------------------标签保持--------------------------------------------------------
    indx = tf.range(FLAGS.batch_size)

    # 随机取batchsize个作为anchor
    # indx_batchsize = tf.range(FLAGS.batch_size)
    # indx = tf.random_shuffle(tf.range(tf.shape(labels)[0]))
    # indx = tf.gather(indx, indx_batchsize, axis=0)


    dist_ap = tf.gather(D_Positive, indx, axis=0)
    # dist_ap = tf.gather(dist_ap, indx, axis=1)

    V_pos = tf.reduce_min(dist_ap, axis=1)   # hardest positive
    I_pos = tf.arg_min(dist_ap, 1)

    # 合成样本也参与
    # dist_ap = tf.gather(D_Positive, indx, axis=0)  # 18,54
    # V_pos = tf.reduce_max(dist_ap, 1)

    Mask_not_drop_pos = (V_pos > 0)

    ############################################
    # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
    #############################################
    D_Negative = Dist
    # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
    D_Negative = tf.where(same, 0 - dis_mask, D_Negative)  # 54 x 54

    # 取最难点要包括合成点
    # num_group = tf.shape(labels)[0] // FLAGS.batch_size
    # D_Negative = tf.reduce_max(D_Negative, 1)  # 54
    # dist_an = tf.reshape(D_Negative, [num_group, FLAGS.batch_size])  # <class 'tuple'>: (3, 18)  Input to reshape is a tensor with 2916 values
    # # dist_an = F.min(temp, axis=0)  # include synthetic positives  包含合成的点
    # # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    # V_neg = tf.reduce_max(dist_an, 0)  # 54


    dist_an = tf.gather(D_Negative, indx, axis=0)  # 18,54
    # D_Negative = tf.reduce_max(D_Negative, 1)  # 18
    # dist_an = tf.reshape(D_Negative, [num_group, FLAGS.batch_size])
    V_neg = tf.reduce_max(dist_an, 1)    # hardest negative

    # I_neg = tf.arg_max(D_Negative, 1)
    # I_neg [9 10 14 13 15 10 13 16 15  0  1  4  0  3  2  4  5  3]
    # I_pos [ 3  6  6  6  8  7  3  5  7 17 17 16 14 17 12 12 11  9]

    # V_pos = tf.Print(V_pos, [V_pos], message='正对相似度', summarize=1000)
    # V_neg = tf.Print(V_neg, [V_neg], message='负对相似度', summarize=1000)
    # I_neg = tf.Print(I_neg, [I_neg], message='负对索引', summarize=1000)
    # I_pos = tf.Print(I_pos, [I_pos], message='正对索引', summarize=1000)

    # prevent duplicated pairs  # 排除样本与样本本身组成的pair

    Mask_not_drop_neg = (V_neg > 0)
    #
    # # extracting neg score  提取位置上的相似度
    # Neg_log = V_neg

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl = tf.stack([V_pos, V_neg], 1)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
    Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg
    # Mask_not_drop = tf.Print(Mask_not_drop, [Mask_not_drop], message='不丢弃：', summarize=1000)

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # logits = trpl / 0.11
    # margin = 1/(5+tf.exp(Jm_loss))
    # Jm_loss = tf.clip_by_value(Jm_loss, 0, 0.8)
    # margin = -Jm_loss/4+0.3
    # margin = tf.Print(margin, [margin], message='间距margin', summarize=1000)


    logits = trpl / temperate
    # logits = tf.Print(logits, [logits], message='损失logits', summarize=1000)
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN


    # Prob = tf.Print(Prob, [Prob], message='log_softmax损失', summarize=1000)
    pro_mask = tf.zeros_like(Prob)
    trip_less_zero = tf.where(Mask_not_drop, Prob, pro_mask)
    # trip_less_zero = tf.Print(trip_less_zero, [trip_less_zero], message='剔除后的损失', summarize=1000)
    label_preserving_loss = tf.reduce_mean(trip_less_zero)  ##

    return loss + 0.1*label_preserving_loss

    # return loss



def hard_neg_mining_epnh_synstage_loss(l2norm_emb, labels, temperate,
                                     reg_lambda=3e-3, print_losses=False):
    # 2份合成样本，每份由anchor、positive、negative组成
    # 现在重新组成anchor、positive、negative的形式
    anchor1, positive1, negative1, anchor2, positive2, negative2 = tf.split(l2norm_emb,6, axis=0)
    anchor = tf.concat([anchor1,anchor2], axis=0)
    positive = tf.concat([positive1,positive2], axis=0)
    negative = tf.concat([negative1,negative2], axis=0)

    # Get per pair similarities.
    # 这儿计算anchors和positives.T的内积，度量两两向量之间的距离
    # 每一行表示一个anchor与每个positives矩阵的元素内积的结果
    similarity_matrix_ap = math_ops.matmul(
        anchor, positive, transpose_a=False,
        transpose_b=True)
    similarity_matrix_an = math_ops.matmul(
        anchor, negative, transpose_a=False,
        transpose_b=True)
    easy_positive = tf.reduce_max(similarity_matrix_ap, axis=1)
    hard_negative = tf.reduce_max(similarity_matrix_an, axis=1)
    # loss = tf.reduce_mean(-tf.log(tf.exp(easy_positive) / (easy_positive + hard_negative)))
    trpl = tf.stack([easy_positive, hard_negative], 1)
    logits = trpl / 0.1
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]
    loss = tf.reduce_mean(Prob)
    return loss

def easy_positive_semi_hard_negative_loss(embeddings, labels,
                reg_lambda=3e-3, print_losses=False):
    # 正则化与其损失
    # reg_embeddings = math_ops.reduce_mean(
    #     math_ops.reduce_sum(math_ops.square(embeddings), 1))
    # l2loss = math_ops.multiply(
    #     0.25 * reg_lambda, reg_embeddings, name='l2loss')

    same, Diff = Mat(labels)

    # embeddings = tf.Print(embeddings, [embeddings], message='归一化前：', summarize=1000)
    l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)
    # l2norm_emb = tf.Print(l2norm_emb, [l2norm_emb], message='l2归一化后：', summarize=1000)
    # Get per pair similarities.
    Dist = distMC(l2norm_emb, l2norm_emb)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)


    ############################################
    # finding max similarity on same label pairs寻找相同标签 相似度最大的pairs
    #############################################
    D_Positive = Dist  #  tf.stop_gradient??
    # 把异类对的距离设为最大
    # D_Positive[Diff] = -1
    dis_mask = tf.ones_like(D_Positive)
    D_Positive = tf.where(Diff, 0-dis_mask, D_Positive)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)
    # D_Positive = tf.multiply(tf.cast(same, tf.float32), D_Positive) - tf.cast(Diff, tf.float32)

    # D_Positive[D_Positive > 0.9999] = -1  阈值为0.9
    D_Positive = tf.where(tf.greater(D_Positive, 0.999), 0 - dis_mask, D_Positive)  # 把正对相似度太大的设为-1， 即不选取太相似的正对，以免loss为0
    # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号

    V_pos  = tf.reduce_max(D_Positive, axis=1)
    I_pos = tf.arg_max(D_Positive, 1)

    Mask_not_drop_pos = (V_pos > 0)

    # extracting pos score  提取位置上的相似度
    Pos_log = V_pos

    ############################################
    # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
    #############################################
    D_Negative = Dist
    # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
    D_Negative = tf.where(same, 0 - dis_mask, D_Negative)
    if FLAGS.semi:  # 是否采取semi hard negative采样
        # choose a anchor-negative pair that is father than
        # the anchor-positive pair, but within a margin
        N = tf.shape(embeddings)[0]
        # pp = V_pos.repeat(N, 1).t()
        # V_pos = tf.Print(V_pos, [V_pos], message='V_pos相似度', summarize=1000)
        pp = tf.transpose(tf.tile(tf.expand_dims(V_pos, axis=0), multiples=[N, 1]))
        # pp = tf.Print(pp, [pp], message='掩码', summarize=1000)
        # mask_semi = (D_Negative > pp) & Diff & (D_Negative < pp + 0.9)
        mask_semi = Diff & (D_Negative > pp + 0.0)
        D_Negative = tf.where(mask_semi, 0 - dis_mask, D_Negative) # 将mask里的设为-1,是要舍弃的样本
        # D_detach_N[(D_detach_N > (pp)) & Diff] = -1  # extracting SHN
    # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    V_neg = tf.reduce_max(D_Negative, 1)
    I_neg = tf.arg_max(D_Negative, 1)
    # I_neg [9 10 14 13 15 10 13 16 15  0  1  4  0  3  2  4  5  3]
    # I_pos [ 3  6  6  6  8  7  3  5  7 17 17 16 14 17 12 12 11  9]

    # prevent duplicated pairs  # 排除样本与样本本身组成的pair

    Mask_not_drop_neg = (V_neg > 0)
    #
    # # extracting neg score  提取位置上的相似度
    # Neg_log = V_neg

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl = tf.stack([V_pos, V_neg], 1)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
    Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    logits = trpl / 0.1
    # logits = tf.Print(logits, [logits], message='概率logits', summarize=1000)
    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # Prob = tf.Print(Prob, [Prob], message='概率Prob', summarize=1000)
    pro_mask = tf.zeros_like(Prob)
    loss = tf.reduce_mean(tf.where(Mask_not_drop, Prob, pro_mask))

    # if print_losses:
    #     xent_loss = logging_ops.Print(
    #         xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
    # r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加

    # r_loss = tf.cond(tf.is_nan(loss), lambda: tf.constant(0.), lambda: loss)  # 加
    positive_emb = tf.gather(embeddings, I_pos)
    negative_emb = tf.gather(embeddings, I_neg)
    positive_label = tf.gather(labels, I_pos)
    negative_label = tf.gather(labels, I_neg)
    return loss, I_pos, positive_emb, I_neg, negative_emb, positive_label, negative_label


# def epnh_synstage_loss(embeddings, reg_lambda=3e-3, print_losses=False):
#     # # 正则化与其损失
#     # reg_embeddings = math_ops.reduce_mean(
#     #     math_ops.reduce_sum(math_ops.square(embeddings), 1))
#     # l2loss = math_ops.multiply(
#     #     0.25 * reg_lambda, reg_embeddings, name='l2loss')
#
#     l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)
#
#     embeddings_split = tf.split(l2norm_emb, 3, axis=0)
#     anchor = embeddings_split[0]
#     positive = embeddings_split[1]
#     negative = embeddings_split[2]
#
#     # 计算anchor-->positive 和 anchor-->negative的距离
#     dis_ap = dist_calculate(anchor, positive)
#     dis_an = dist_calculate(anchor, negative)
#     # dis_ap = tf.Print(dis_ap, [dis_ap], message='三元组dis_ap：', summarize=1000)
#     # dis_an = tf.Print(dis_an, [dis_an], message='三元组dis_an：', summarize=1000)
#
#     # triplets  ###############根据提取的正对负对构建三元组#########################
#     trpl = tf.stack([dis_ap, dis_an], 1)
#     # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
#
#     # loss triplet_L=max(0, dap-dan+margin)
#     # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
#     # margin = 1 / (8 + tf.exp(-Jm_loss))
#     # margin = 1 / (5 + tf.exp(Jm_loss))
#     # Jm_loss = tf.clip_by_value(Jm_loss, 0, 0.8)
#     # margin = -Jm_loss / 4 + 0.3
#     logits = trpl / 0.4
#
#     Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
#     # Prob = tf.Print(Prob, [Prob], message='概率', summarize=1000)
#     loss = tf.reduce_mean(Prob)
#
#     # r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加
#     # r_loss = tf.cond(tf.is_nan(loss), lambda: tf.constant(0.), lambda: loss)
#     return loss
#
#     # # l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)
#     # embeddings_split = tf.split(embeddings, 3, axis=0)
#     # anchor = embeddings_split[0]
#     # positive = embeddings_split[1]
#     # negative = embeddings_split[2]
#     # d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
#     # d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
#     # loss = tf.maximum(d_ap - d_an + 1.0, 0.)
#     # return tf.reduce_sum(loss)


def epnh_synstage_loss(embeddings, reg_lambda=3e-3, print_losses=False):
    # # 正则化与其损失
    # reg_embeddings = math_ops.reduce_mean(
    #     math_ops.reduce_sum(math_ops.square(embeddings), 1))
    # l2loss = math_ops.multiply(
    #     0.25 * reg_lambda, reg_embeddings, name='l2loss')

    l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)

    embeddings_split = tf.split(l2norm_emb, 3, axis=0)
    anchor = embeddings_split[0]
    positive = embeddings_split[1]
    negative = embeddings_split[2]

    # 计算anchor-->positive 和 anchor-->negative的距离
    dis_ap = dist_calculate(anchor, positive)
    dis_an = dist_calculate(anchor, negative)
    # dis_ap = tf.Print(dis_ap, [dis_ap], message='三元组dis_ap：', summarize=1000)
    # dis_an = tf.Print(dis_an, [dis_an], message='三元组dis_an：', summarize=1000)
    Mask_not_drop_pos = (dis_ap > 0)
    Mask_not_drop_neg = (dis_an > 0)
    Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl = tf.stack([dis_ap, dis_an], 1)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # margin = 1 / (8 + tf.exp(-Jm_loss))
    # margin = 1 / (5 + tf.exp(Jm_loss))
    # Jm_loss = tf.clip_by_value(Jm_loss, 0, 0.8)
    # margin = -Jm_loss / 4 + 0.3
    logits = trpl / 0.1

    Prob = -tf.nn.log_softmax(logits, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # Prob = tf.Print(Prob, [Prob], message='概率', summarize=1000)

    pro_mask = tf.zeros_like(Prob)
    trip_less_zero = tf.where(Mask_not_drop, Prob, pro_mask)
    loss = tf.reduce_mean(trip_less_zero)

    # r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加
    # r_loss = tf.cond(tf.is_nan(loss), lambda: tf.constant(0.), lambda: loss)
    return loss

    # # l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)
    # embeddings_split = tf.split(embeddings, 3, axis=0)
    # anchor = embeddings_split[0]
    # positive = embeddings_split[1]
    # negative = embeddings_split[2]
    # d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    # d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    # loss = tf.maximum(d_ap - d_an + 1.0, 0.)
    # return tf.reduce_sum(loss)


def easy_positive_n_hard_negative_loss(embeddings, labels,
                reg_lambda=3e-3, print_losses=False):
    # 正则化与其损失
    reg_embeddings = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_embeddings, name='l2loss')

    same, Diff = Mat(labels)

    l2norm_emb = tf.nn.l2_normalize(embeddings, dim=1)

    # Get per pair similarities.
    Dist = distMC(l2norm_emb, l2norm_emb)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)


    ############################################
    # finding max similarity on same label pairs寻找相同标签 相似度最大的pairs
    #############################################
    D_Positive = Dist  #  tf.stop_gradient??
    # 把异类对的距离设为最大
    # D_Positive[Diff] = -1
    dis_mask = tf.ones_like(D_Positive)
    D_Positive = tf.where(Diff, 0-dis_mask, D_Positive)
    # D_Positive = tf.Print(D_Positive, [D_Positive, dis_mask, Diff], message='Debug message:', summarize=1000)
    # D_Positive = tf.multiply(tf.cast(same, tf.float32), D_Positive) - tf.cast(Diff, tf.float32)

    # D_Positive[D_Positive > 0.9999] = -1
    D_Positive = tf.where(tf.greater(D_Positive, 0.9), 0 - dis_mask, D_Positive)
    # 找到各样本最近的样本（同标签）， 返回最大相似度、与在矩阵中的列号
    V_pos  = tf.reduce_max(D_Positive, axis=1)
    # I_pos = tf.arg_max(D_Positive, 1)
    V_pos_mask = tf.stack([V_pos, V_pos], 1)
    Mask_not_drop_pos = (V_pos_mask > 0)

    # extracting pos score  提取位置上的相似度
    Pos_log = V_pos

    ############################################
    # finding max similarity on diff label pairs  寻找相似度最大的不同标签对
    #############################################
    D_Negative = Dist
    # D_Negative[same] = -1  # 把标签相同的pair的相似度设为-1
    D_Negative = tf.where(same, 0 - dis_mask, D_Negative)
    if FLAGS.semi:  # 是否采取semi hard negative采样
        # choose a anchor-negative pair that is father than
        # the anchor-positive pair, but within a margin
        N = tf.shape(embeddings)[0]
        # pp = V_pos.repeat(N, 1).t()
        # V_pos = tf.Print(V_pos, [V_pos], message='V_pos相似度', summarize=1000)
        pp = tf.transpose(tf.tile(tf.expand_dims(V_pos, axis=0), multiples=[N, 1]))
        # pp = tf.Print(pp, [pp], message='掩码', summarize=1000)
        mask_semi = (D_Negative > pp) & Diff
        D_Negative = tf.where(mask_semi, 0 - dis_mask, D_Negative)
        # D_detach_N[(D_detach_N > (pp)) & Diff] = -1  # extracting SHN
    # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    # V_neg = tf.reduce_max(D_Negative, 1)
    # # I_neg = tf.arg_max(D_Negative, 1)
    # pp = tf.transpose(tf.tile(tf.expand_dims(V_neg, axis=0), multiples=[N, 1]))
    # mask_max1 = (D_Negative > pp)
    # D_Negative = tf.where(mask_max1, 0 - dis_mask, D_Negative)
    # # 第二大
    # V_neg2 = tf.reduce_max(D_Negative, 1)

    V_neg= tf.nn.top_k(D_Negative, 2).values
    # V_neg = tf.Print(V_neg, [V_neg], message='V_neg', summarize=1000)
    V_neg1, V_neg2 = tf.split(V_neg, 2, axis=1)
    # V_neg1 = tf.Print(V_neg1, [V_neg1], message='V_neg1', summarize=1000)
    # V_neg2 = tf.Print(V_neg2, [V_neg2], message='V_neg2', summarize=1000)
    # V_neg3 = tf.Print(V_neg3, [V_neg3], message='V_neg3', summarize=1000)
    V_neg1 = tf.reshape(V_neg1, [tf.shape(V_neg1)[0]])
    # V_neg1 = tf.Print(V_neg1, [V_neg1], message='V_neg1_reshape', summarize=1000)
    V_neg2 = tf.reshape(V_neg2, [tf.shape(V_neg2)[0]])
    # V_neg3 = tf.reshape(V_neg3, [tf.shape(V_neg3)[0]])

    # prevent duplicated pairs  # 排除样本与样本本身组成的pair
    V_neg_mask = tf.stack([V_neg1, V_neg2], 1)
    Mask_not_drop_neg = (V_neg_mask > 0)
    #
    # # extracting neg score  提取位置上的相似度
    # Neg_log = V_neg

    # triplets  ###############根据提取的正对负对构建三元组#########################
    trpl1 = tf.stack([V_pos, V_neg1], 1)
    trpl2 = tf.stack([V_pos, V_neg2], 1)
    # trpl3 = tf.stack([V_pos, V_neg3], 1)
    trpl = tf.concat([trpl1, trpl2], 0)
    # trpl = tf.Print(trpl, [trpl], message='三元组：', summarize=1000)
    # Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg

    # loss triplet_L=max(0, dap-dan+margin)
    # Prob = tf.nn.log_softmax(V_neg / V_pos, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    Prob = -tf.nn.log_softmax(trpl / 0.1, dim=1)[:, 0]  # 16对。。。 log(M/N) = logM-logN
    # Prob = tf.Print(Prob, [Prob], message='概率', summarize=1000)
    # pro_mask = tf.zeros_like(Prob)
    # loss = tf.reduce_mean(tf.where(Mask_not_drop, Prob, pro_mask))
    loss = tf.reduce_mean(Prob)

    # if print_losses:
    #     xent_loss = logging_ops.Print(
    #         xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
    r_loss = tf.cond(tf.is_nan(loss), lambda: tf.constant(0.), lambda: loss)  # 加
    # r_loss = tf.cond(tf.is_nan(loss + l2loss), lambda: tf.constant(0.), lambda: loss + l2loss)  # 加
    return r_loss


def lifted_struct_loss(labels, embeddings, margin=1.0):
    """Computes the lifted structured loss.
      The loss encourages the positive distances (between a pair of embeddings
      with the same labels) to be smaller than any negative distances (between a
      pair of embeddings with different labels) in the mini-batch in a way
      that is differentiable with respect to the embedding vectors.
      See: https://arxiv.org/abs/1511.06452.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        lifted_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = pairwise_distance(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    diff = margin - pairwise_distances
    mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = math_ops.reduce_min(diff, 1, keep_dims=True)
    row_negative_maximums = math_ops.reduce_max(
        math_ops.multiply(
            diff - row_minimums, mask), 1, keep_dims=True) + row_minimums

    max_elements = math_ops.maximum(
        row_negative_maximums, array_ops.transpose(row_negative_maximums))
    diff_tiled = array_ops.tile(diff, [batch_size, 1])
    mask_tiled = array_ops.tile(mask, [batch_size, 1])
    max_elements_vect = array_ops.reshape(
        array_ops.transpose(max_elements), [-1, 1])

    loss_exp_left = array_ops.reshape(
        math_ops.reduce_sum(math_ops.multiply(
            math_ops.exp(
                diff_tiled - max_elements_vect),
            mask_tiled), 1, keep_dims=True), [batch_size, batch_size])

    loss_mat = max_elements + math_ops.log(
        loss_exp_left + array_ops.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = math_ops.reduce_sum(mask_positives) / 2.0

    lifted_loss = math_ops.truediv(
        0.25 * math_ops.reduce_sum(
            math_ops.square(
                math_ops.maximum(
                    math_ops.multiply(loss_mat, mask_positives), 0.0))),
        num_positives,
        name='liftedstruct_loss')
    return lifted_loss


def Binary_sofmax(anc_emb, pos_emb, label, cycle, weight, emb_size):
    cross_entropy = tf.constant(0., tf.float32)
    with tf.variable_scope("Softmax_classifier"):
        W_fc = nn_Ops.weight_variable([2048, 2], "softmax_w", wd=False)
        b_fc = nn_Ops.bias_variable([2], "softmax_b")
    for i in range(cycle):
        if i >= 64:
            break
        pos_f = tf.slice(input_=pos_emb, begin=[0, 0], size=[i, emb_size])
        label_f = tf.slice(input_=label, begin=[0], size=[i])
        pos_b = tf.slice(input_=pos_emb, begin=[i, 0], size=[64 - i, emb_size])
        label_b = tf.slice(input_=label, begin=[i], size=[64 - i])
        pos_temp = tf.concat([pos_b, pos_f], axis=0)
        label_temp = tf.concat([label_b, label_f], axis=0)
        logits = tf.matmul(tf.concat([anc_emb, pos_temp], axis=1), W_fc) + b_fc
        label_binary = tf.cast(tf.equal(label, label_temp), tf.int32)
        weight_m = tf.cast(tf.logical_not(tf.equal(label, label_temp)), tf.float32) \
                   * weight + tf.cast(label_binary, tf.float32)
        cross_entropy += tf.reduce_mean(
            tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels=label_binary), weight_m))/np.float32(cycle)
    return cross_entropy, W_fc, b_fc


def Loss(embedding, label, temperate, _lossType="Softmax", loss_l2_reg=FLAGS.loss_l2_reg):
    """
    选择调用各损失函数
    :param embedding: 嵌入z  (batch_size, 128)
    :param label: 标签 (batch_size)
    :param _lossType:  损失函数类型，默认为Softmax（交叉熵损失）
    :param loss_l2_reg:  嵌入l2_loss的因子 0.003
    :return: 损失
    """
    # 将一个batch内的嵌入和标签都平均划分成两个子集合
    embedding_split = tf.split(embedding, 2, axis=0)
    label_split = tf.split(label, 2, axis=0)
    # 构建anchor和positive及标签
    embedding_anchor = embedding_split[0]
    embedding_positive = embedding_split[1]
    label_positive = label_split[1]
    # 初始化loss 为0
    _Loss = 0

    if _lossType == "Softmax":
        print("Use Softmax")
        W_fc2 = nn_Ops.weight_variable([FLAGS.embedding_size, FLAGS.num_class], "softmax_w")
        b_fc2 = nn_Ops.bias_variable([FLAGS.num_class], "softmax_b")
        # W_fc2 = nn_Ops.weight_variable([1024, 10], 'softmax_w')
        # b_fc2 = nn_Ops.bias_variable([10], 'softmax_b')
        y_conv = tf.matmul(embedding, W_fc2) + b_fc2
        _Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=y_conv))
        # _Loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_conv))

    elif _lossType == "Contrastive_Loss_v2":
        print("Use Contrastive_Loss_v2")
        embedding_anchor = tf.split(embedding_anchor, 2, axis=0)
        embedding_positive = tf.split(embedding_positive, 2, axis=0)
        _Loss = contrastive_loss_v2(
            embedding_anchor[0], embedding_anchor[1], embedding_positive[0], embedding_positive[1], alpha=1.0)

    elif _lossType == "Contrastive_Loss":
        print("Use Contrastive_Loss")
        embedding3 = tf.split(embedding, 2, axis=0)
        positive = embedding3[0]
        negative = embedding3[1]

        # 是否同一个标签
        label1, label2 = tf.split(label, 2, axis=0)
        # label1 = tf.Print(label1, [label1], "label1", summarize=100)
        # label2 = tf.Print(label2, [label2], "label2", summarize=100)
        label_ifsame1 = tf.bitwise.bitwise_xor(label1, label2)
        label_ifsame = tf.ones(tf.shape(label_ifsame1),tf.int32) - label_ifsame1
        # label_ifsame = tf.Print(label_ifsame, [label_ifsame], "label_ifsame", summarize=100)
        _Loss = contrastive_loss(label_ifsame,
            positive,  negative, margin=1.0)

    elif _lossType == "Triplet_Semihard":
        print("Use Triplet_semihard")
        _Loss = triplet_semihard_loss(label, embedding)

    elif _lossType == "LiftedStructLoss":
        print("Use LiftedStructLoss")
        _Loss = lifted_struct_loss(label, embedding)

    elif _lossType == "NpairLoss":
        print("Use NpairLoss")
        _Loss = npairs_loss(label_positive, embedding_anchor, embedding_positive, reg_lambda=loss_l2_reg)

    elif _lossType == "Triplet":
        print("Use Triplet Loss")
        embedding3 = tf.split(embedding, 3, axis=0)
        anchor = embedding3[0]
        positive = embedding3[1]
        negative = embedding3[2]
        _Loss = triplet_loss(anchor, positive, negative)
        
    elif _lossType == "New_npairLoss":
        print("Use new NpairLoss")
        _Loss = new_npair_loss(
            labels=label, embedding_anchor=embedding_anchor,
            embedding_positive=embedding_positive, reg_lambda=loss_l2_reg,
            equal_shape=True, half_batch_size=int(FLAGS.batch_size/2))

    elif _lossType == "easy_pos_hard_negLoss":
        print("Use easy_pos_hard_negLoss")
        embedding = tf.nn.l2_normalize(embedding, axis=1)
        anchor, positive, negative = tf.split(embedding, 3, axis=0)
        _Loss = easy_positive_hard_negative_loss(anchor, positive, negative)
        return _Loss

    elif _lossType == "epnh_synstage_loss":
        _Loss = epnh_synstage_loss(embedding)
        return _Loss

    elif _lossType == "hard_neg_mining_epnh_synstage_loss":
        embedding = tf.nn.l2_normalize(embedding, axis=1)
        _Loss = hard_neg_mining_epnh_synstage_loss(embedding, label, temperate)
        return _Loss

    # elif _lossType == "hard_neg_mining_epnh_synstage_loss":
    #     _Loss = label_preserving_loss(embedding, label, temperate)
    #     return _Loss

    elif _lossType == "ephn_label_preserving_loss":   # fdddddddddd
        _Loss = ephn_label_preserving_loss(embedding, label, temperate)
        return _Loss

    elif _lossType == "easy_pos_semi_hard_negLoss":

        print("Use easy_pos_semi_hard_negLoss")

        _Loss, _I_pos, _positive_emb, _I_neg, _negative_emb, _positive_label, _negative_label \
            = easy_positive_semi_hard_negative_loss(embedding, label)

        return _Loss, _I_pos, _positive_emb, _I_neg, _negative_emb, _positive_label, _negative_label


        # 一个正例对应两个难样本
        # _Loss = easy_positive_n_hard_negative_loss(
        #     embedding, label)

    return _Loss



