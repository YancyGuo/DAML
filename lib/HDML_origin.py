from flags.FLAGS_HDML_triplet import *
from lib.nn_Ops import distance, weight_variable, bias_variable
import os


# def Pulling(Loss_type, embedding, Jm):
#     """
#
#     :param Loss_type:
#     :param embedding:
#     :param Jm:  上一个epoch的平均度量损失·1 ，1·
#     :return:
#     """
#     if Loss_type == 'NpairLoss':
#         # 构建anchor pos neg
#         embedding_split = tf.split(embedding, 2, axis=0)
#         anc = embedding_split[0]
#         pos = embedding_split[1]
#         neg = pos
#
#         anc_tile = tf.reshape(tf.tile(anc, [1, int(FLAGS.batch_size / 2)]), [-1, int(FLAGS.embedding_size)])
#         pos_tile = tf.reshape(tf.tile(pos, [1, int(FLAGS.batch_size / 2)]), [-1, int(FLAGS.embedding_size)])
#         neg_tile = tf.tile(neg, [int(FLAGS.batch_size / 2), 1])
#
#         # 插值生成Z^-
#         neg2_tile = anc_tile + tf.multiply(
#             (neg_tile - anc_tile),
#             tf.tile(
#                 ((distance(anc_tile, pos_tile) + (distance(anc_tile, neg_tile) - distance(anc_tile, pos_tile)) * tf.exp(
#                     -FLAGS.alpha / Jm)) / distance(anc_tile, neg_tile)), [1, int(FLAGS.embedding_size)]
#             )
#         )
#         # if D_ap is larger than D_an, the element will be True(1) 设置开关
#         # 当d(z,z-)<=d+时， Z^- = Z-
#         neg_mask = tf.greater_equal(distance(anc_tile, pos_tile), distance(anc_tile, neg_tile))
#         op_neg_mask = tf.logical_not(neg_mask)  # 逻辑非
#         neg_mask = tf.cast(neg_mask, tf.float32)
#         op_neg_mask = tf.cast(op_neg_mask, tf.float32)
#
#         # 论文公式7，这里相当于一个switch
#         neg_tile = tf.multiply(neg_tile, neg_mask) + tf.multiply(neg2_tile, op_neg_mask)
#         embedding_z_quta = tf.concat([anc, neg_tile], axis=0)
#         return embedding_z_quta
#
#     elif Loss_type == 'Triplet':
#         embedding_split = tf.split(embedding, 3, axis=0)  # （？，256）
#         anc = embedding_split[0]  # （？，256）
#         pos = embedding_split[1]  # （？，256）
#         neg = embedding_split[2]  # （？，256）
#
#         # anc = tf.Print(anc, [anc], "anc")
#         # pos = tf.Print(pos, [pos], "pos")
#         # neg = tf.Print(neg, [neg], "neg")
#         #
#         # # 论文公式7
#         # Jm = tf.Print(Jm, [Jm], "Jm")
#         # lam = tf.exp(-FLAGS.alpha / Jm) + 0.03
#         # # lam = tf.Print(lam, [lam], "lam")
#         # dis_a_n = distance(anc, neg)
#         # # dis_a_n = tf.Print(dis_a_n, [dis_a_n], "dis_a_n")
#         # dis_a_p = distance(anc, pos)
#         # # dis_a_p = tf.Print(dis_a_p, [dis_a_p], "dis_a_p")
#         # neg2 = anc + tf.multiply(
#         #     (neg - anc),
#         #     tf.tile(
#         #         ((dis_a_p + (dis_a_n - dis_a_p) * lam) / dis_a_n), [1, int(FLAGS.embedding_size)]
#         #     )
#         # )
#
#         # neg2 = anc + tf.multiply(
#         #     (neg - anc),
#         #     tf.tile(
#         #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * (tf.exp(
#         #             -FLAGS.alpha / Jm) + 1e-2)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
#         #     )
#         # )
#
#         # neg2 = anc + tf.multiply(
#         #     (neg - anc),
#         #     tf.tile(
#         #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
#         #             -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
#         #     )
#         # )
#         # neg2 = tf.Print(neg2, [neg2], "neg2", summarize=10)
#
#         # neg2 = anc + tf.multiply(
#         #     (anc - neg),
#         #     tf.tile(
#         #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
#         #             -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
#         #     )
#         # )
#
#         neg2 = anc + 0.2*(anc - neg)   # 02-09-12-19
#
#         # if D_ap is larger than D_an, the element will be True(1)
#         # 如果d(z,z-)>d+,则生成样本为neg2, 否则为neg本身
#         neg_mask = tf.greater_equal(distance(anc, pos), distance(anc, neg))
#         op_neg_mask = tf.logical_not(neg_mask)
#         neg_mask = tf.cast(neg_mask, tf.float32)
#         op_neg_mask = tf.cast(op_neg_mask, tf.float32)
#         # neg = tf.multiply(neg, neg_mask) + tf.multiply(neg2, op_neg_mask)
#         neg = tf.multiply(neg, neg_mask) + tf.multiply(neg2, op_neg_mask)
#
#         # embedding2 is the pulled embedding
#         embedding_z_quta = tf.concat([anc, pos, neg], axis=0)
#         return embedding_z_quta
#
#     else:
#         print("Your loss type is not suit for HDML")
#         os._exit()


def Pulling(Loss_type, embedding, Jm):
    """

    :param Loss_type:
    :param embedding:
    :param Jm:  上一个epoch的平均度量损失·1 ，1·
    :return:
    """
    if Loss_type == 'NpairLoss':
        # 构建anchor pos neg
        embedding_split = tf.split(embedding, 2, axis=0)  # （FLAGS.batch_size， 128）
        anc = embedding_split[0]  # （FLAGS.batch_size / 2，128）
        pos = embedding_split[1]  # (FLAGS.batch_size / 2, 128) 即（8， 128）
        neg = pos

        # anc 和 pos均在第二维度复制8次，然后reshape成 （64，128）
        tmp = tf.tile(anc, [1, int(FLAGS.batch_size / 2)])  # Tensor("Tile:0", shape=(?, 1024), dtype=float32)
        anc_tile = tf.reshape(tmp,
                              [-1, int(FLAGS.embedding_size)])  # Tensor("Reshape:0", shape=(64, 128), dtype=float32)
        # Tensor("Reshape_1:0", shape=(64, 128), dtype=float32)
        pos_tile = tf.reshape(tf.tile(pos, [1, int(FLAGS.batch_size / 2)]), [-1, int(FLAGS.embedding_size)])
        # Tensor("Tile_2:0", shape=(？, 128), dtype=float32)  （64， 128） # 在第一维度复制8次
        neg_tile = tf.tile(neg, [int(FLAGS.batch_size / 2), 1])

        dis_ap = distance(anc_tile, pos_tile)  # (64,1)
        # 插值生成Z^-
        neg2_tile = anc_tile + tf.multiply(
            (tf.exp(-FLAGS.alpha / Jm) * distance(anc_tile, neg_tile)
             + (1 - tf.exp(-FLAGS.alpha / Jm) * distance(anc_tile, pos_tile))),
            (neg_tile - anc_tile) / distance(anc_tile, neg_tile)
        )

        # if D_ap is larger than D_an, the element will be True(1) 设置开关
        # 当d(z,z-)<=d+时， Z^- = Z-
        neg_mask = tf.greater_equal(distance(anc_tile, pos_tile), distance(anc_tile, neg_tile))
        op_neg_mask = tf.logical_not(neg_mask)  # 逻辑非
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)

        # 论文公式7，这里相当于一个switch
        neg_tile = tf.multiply(neg_tile, neg_mask) + tf.multiply(neg2_tile, op_neg_mask)
        embedding_z_quta = tf.concat([anc, neg_tile], axis=0)  # (72,128)  3 anchor还是原来的，后面会split出来，其余64个作为neg

        return embedding_z_quta
            # , pos_tile, neg_tile, neg2_tile, dis_ap
        # return embedding_z_quta, pos, neg, neg2

    elif Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)  # （？，256）
        anc = embedding_split[0]  # （？，256）
        pos = embedding_split[1]  # （？，256）
        neg = embedding_split[2]  # （？，256）

        # anc = tf.Print(anc, [anc], "anc")
        # pos = tf.Print(pos, [pos], "pos")
        # neg = tf.Print(neg, [neg], "neg")
        #
        # # 论文公式7
        # Jm = tf.Print(Jm, [Jm], "Jm")
        # lam = tf.exp(-FLAGS.alpha / Jm) + 0.03
        # # lam = tf.Print(lam, [lam], "lam")
        # dis_a_n = distance(anc, neg)
        # # dis_a_n = tf.Print(dis_a_n, [dis_a_n], "dis_a_n")
        # dis_a_p = distance(anc, pos)
        # # dis_a_p = tf.Print(dis_a_p, [dis_a_p], "dis_a_p")
        # neg2 = anc + tf.multiply(
        #     (neg - anc),
        #     tf.tile(
        #         ((dis_a_p + (dis_a_n - dis_a_p) * lam) / dis_a_n), [1, int(FLAGS.embedding_size)]
        #     )
        # )

        # neg2 = anc + tf.multiply(
        #     (neg - anc),
        #     tf.tile(
        #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * (tf.exp(
        #             -FLAGS.alpha / Jm) + 1e-2)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
        #     )
        # )

        # neg2 = anc + tf.multiply(
        #     (neg - anc),
        #     tf.tile(
        #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
        #             -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
        #     )
        # )
        # neg2 = tf.Print(neg2, [neg2], "neg2", summarize=10)
        #
        # neg2 = anc + tf.multiply(
        #     (anc - neg),
        #     tf.tile(
        #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
        #             -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
        #     )
        # )

        # neg2 = capsule(embedding)  # 调用动态路由算法

        neg2 = neg + tf.multiply(
            (anc - neg),
            tf.tile(
                ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
                    -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
            )
        )

        # neg = tf.Print(neg, [neg], "neg", summarize=1000)
        # neg2 = tf.Print(neg2, [neg2], "neg2", summarize=1000)

        # neg2 = anc + tf.exp(-FLAGS.alpha / Jm)*(neg - anc)
        # neg2 = anc + 0.2*(neg - anc)   # 02-09-12-19

        # if D_ap is larger than D_an, the element will be True(1)
        # 如果d(z,z-)>d+,则生成样本为neg2, 否则为neg本身
        ap = distance(anc, pos)
        # ap = tf.Print(ap, [ap], "ap", summarize=1000)
        an = distance(anc, neg)
        # an = tf.Print(an, [an], "an", summarize=1000)

        neg_mask = tf.greater_equal(ap, an)
        # neg_mask = tf.greater_equal(distance(anc, pos), distance(anc, neg))
        # neg_mask = tf.Print(neg_mask, [neg_mask], "neg_mask", summarize=1000)
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        # neg = tf.multiply(neg, neg_mask) + tf.multiply(neg, op_neg_mask)
        neg = tf.multiply(neg, neg_mask) + tf.multiply(neg2, op_neg_mask)

        # embedding2 is the pulled embedding
        embedding_z_quta = tf.concat([anc, pos, neg], axis=0)
        # tf.summary.histogram('histogram_neg2', neg2)
        # tf.summary.histogram('histogram_neg', neg)
        # return embedding_z_quta, neg2, neg, neg_mask
        return embedding_z_quta

    elif Loss_type == 'NpairLoss_refind':
        # 构建anchor pos neg
        embedding_split = tf.split(embedding, 2, axis=0)  # （FLAGS.batch_size， 128）
        anc = embedding_split[0]  # （FLAGS.batch_size / 2，128）
        pos = embedding_split[1]  # (FLAGS.batch_size / 2, 128) 即（8， 128）
        neg = pos




        # anc 和 pos均在第二维度复制8次，然后reshape成 （64，128）
        tmp = tf.tile(anc, [1, int(FLAGS.batch_size / 2)])  # Tensor("Tile:0", shape=(?, 1024), dtype=float32)
        anc_tile = tf.reshape(tmp,
                              [-1, int(FLAGS.embedding_size)])  # Tensor("Reshape:0", shape=(64, 128), dtype=float32)
        # Tensor("Reshape_1:0", shape=(64, 128), dtype=float32)
        pos_tile = tf.reshape(tf.tile(pos, [1, int(FLAGS.batch_size / 2)]), [-1, int(FLAGS.embedding_size)])
        # Tensor("Tile_2:0", shape=(？, 128), dtype=float32)  （64， 128） # 在第一维度复制8次
        neg_tile = tf.tile(neg, [int(FLAGS.batch_size / 2), 1])

        w1 = weight_variable([FLAGS.embedding_size, FLAGS.embedding_size], name='gen_w1')
        b1 = bias_variable([FLAGS.embedding_size], name='gen_b1')
        b2 = bias_variable([FLAGS.embedding_size], name='gen_b2')
        w2 = weight_variable([FLAGS.embedding_size, FLAGS.embedding_size], name='gen_w2')
        neg2 = tf.matmul(neg_tile, w1) + b1 + \
               tf.matmul(neg_tile - anc_tile, w2) + b2

        # w1 = weight_variable([8, 64],name='gen_w1')
        # b1 = bias_variable([64], name='gen_b1')
        # b2 = bias_variable([64], name='gen_b2')
        # w2 = weight_variable([8, 64], name='gen_w2')
        # neg2 = tf.transpose(tf.matmul(neg, w1,transpose_a=True)+ b1) + \
        #       tf.transpose(tf.matmul(neg-anc, w2,transpose_a=True)+ b2)

        # dis_ap = distance(anc_tile, pos_tile)  # (64,1)
        # # 插值生成Z^-
        # # neg2_tile = anc_tile + tf.multiply(
        # #     (neg_tile - anc_tile),
        # #     tf.tile(
        # #         ((dis_ap + (distance(anc_tile, neg_tile) - distance(anc_tile, pos_tile)) * tf.exp(
        # #             -FLAGS.alpha / Jm)) / distance(anc_tile, neg_tile)), [1, int(FLAGS.embedding_size)]
        # #     )
        # # )
        #
        # # neg2_tile = anc_tile + tf.multiply(
        # #     (tf.exp(-FLAGS.alpha / Jm) * distance(anc_tile, neg_tile)
        # #      + (1 - tf.exp(-FLAGS.alpha / Jm) * distance(anc_tile, pos_tile))),
        # #     (neg_tile - anc_tile) / distance(anc_tile, neg_tile)
        # # )
        # neg2_tile = neg_tile + tf.exp(-FLAGS.alpha / Jm) * (anc_tile - neg_tile)
        # #             + tf.multiply(
        #     (tf.exp(-FLAGS.alpha / Jm) * distance(neg_tile, anc_tile)
        #      + (1 - tf.exp(-FLAGS.alpha / Jm) * distance(anc_tile, pos_tile))),
        #     (anc_tile - neg_tile) / distance(neg_tile, anc_tile)
        # )
        # neg2_tile = anc_tile + tf.multiply(
        #     tf.exp(-FLAGS.alpha / Jm),
        #     (neg_tile - anc_tile)
        # )

        # similarity_matrix = tf.matmul(anc[i], neg2_tile, transpose_a=False, transpose_b=True)
        #
        # # if D_ap is larger than D_an, the element will be True(1) 设置开关
        # # 当d(z,z-)<=d+时， Z^- = Z-
        # neg_mask = tf.greater_equal(distance(anc_tile, pos_tile), distance(anc_tile, neg_tile))
        # op_neg_mask = tf.logical_not(neg_mask)  # 逻辑非
        # neg_mask = tf.cast(neg_mask, tf.float32)
        # op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        #
        # # 论文公式7，这里相当于一个switch
        # neg_tile = tf.multiply(neg_tile, neg_mask) + tf.multiply(neg2_tile, op_neg_mask)
        embedding_z_quta = tf.concat([anc, neg2], axis=0)  # (72,128)  3 anchor还是原来的，后面会split出来，其余64个作为neg

        # triplet的
        # neg2 = neg + tf.multiply(
        #     (anc - neg),
        #     tf.tile(
        #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * tf.exp(
        #             -FLAGS.alpha / Jm)) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
        #     )
        # )
        #
        # ap = distance(anc, pos)
        # an = distance(anc, neg)
        #
        # neg_mask = tf.greater_equal(ap, an)
        # op_neg_mask = tf.logical_not(neg_mask)
        # neg_mask = tf.cast(neg_mask, tf.float32)
        # op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        # # neg = tf.multiply(neg, neg_mask) + tf.multiply(neg, op_neg_mask)
        # neg = tf.multiply(neg, neg_mask) + tf.multiply(neg2, op_neg_mask)
        #
        # # embedding2 is the pulled embedding
        # embedding_z_quta = tf.concat([anc, pos, neg], axis=0)

        return embedding_z_quta
        # return embedding_z_quta, pos, neg, neg2

    elif Loss_type == 'easy_pos_hard_negLoss' or 'easy_pos_semi_hard_negLoss':
        embedding_split = tf.split(embedding, 3, axis=0)  # （？，256）
        anc = embedding_split[0]  # （？，256）
        pos = embedding_split[1]  # （？，256）
        neg = embedding_split[2]  # （？，256）

        # Javg = tf.clip_by_value(Jm, 1e-4, tf.reduce_max(Jm))
        # factor = 1 - tf.exp(
        #             -FLAGS.alpha / Javg)
        # # factor = tf.tanh(
        # #     FLAGS.alpha / Javg)
        # # factor = tf.exp(
        # #     -Jm)
        #
        # neg2 = neg + tf.multiply(
        #     (anc - neg),
        #     tf.tile(
        #         ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * factor) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
        #     )
        # )

        factor = tf.exp(
            -FLAGS.alpha / Jm)
        # lamda = factor + (1-factor)*(distance(anc, pos) / distance(anc, neg))

        neg2 = anc + tf.multiply(
            (neg - anc),
            tf.tile(
                ((distance(anc, pos) + (distance(anc, neg) - distance(anc, pos)) * factor) / distance(anc, neg)), [1, int(FLAGS.embedding_size)]
            )
        )

        # neg = tf.Print(neg, [neg], "neg", summarize=1000)
        # neg2 = tf.Print(neg2, [neg2], "处理前的neg2", summarize=1000)

        # neg2 = anc + tf.exp(-FLAGS.alpha / Jm)*(neg - anc)
        # neg2 = anc + 0.2*(neg - anc)   # 02-09-12-19

        # if D_ap is larger than D_an, the element will be True(1)
        # 如果d(z,z-)>d+,则生成样本为neg2, 否则为neg本身
        ap = distance(anc, pos)
        # ap = tf.Print(ap, [ap], "ap", summarize=1000)
        an = distance(anc, neg)
        # an = tf.Print(an, [an], "an", summarize=1000)

        neg_mask = tf.greater_equal(ap, an)
        # neg_mask = tf.greater_equal(distance(anc, pos), distance(anc, neg))
        # neg_mask = tf.Print(neg_mask, [neg_mask], "neg_mask", summarize=1000)
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        # neg = tf.multiply(neg, neg_mask) + tf.multiply(neg, op_neg_mask)
        neg = tf.multiply(neg, neg_mask) + tf.multiply(neg2, op_neg_mask)
        # neg = tf.Print(neg, [neg], "处理后的neg", summarize=1000)

        # embedding2 is the pulled embedding
        embedding_z_quta = tf.concat([anc, pos, neg], axis=0)
        # tf.summary.histogram('histogram_neg2', neg2)
        # tf.summary.histogram('histogram_neg', neg)
        # return embedding_z_quta, neg2, neg, neg_masks
        return embedding_z_quta, factor

    else:
        print("Your loss type is not suit for HDML")
        os._exit()


def capsule(input):
    ''' 层l+1中的单个胶囊的路由算法。
    参数：
        input： 张量 [batch_size, num_caps_l=1152, length(u_i)=8,  1]
                num_caps_l为l层的胶囊数
        b_IJ:
        idx_j:

    返回：
        张量 [batch_size, 1, length(v_j)=16, 1] 表示
        l+1层的胶囊j的输出向量`v_j`
    注意：
        u_i表示l层胶囊i的输出向量
        v_j则表示l+1层胶囊j的输出向量
    '''

    with tf.variable_scope('routing'):
        # #   ------------------初始化数据 w  和 u预测向量---------------------------------
        # # W_ij[8,16] 因为PrimaryCaps输出的是8D向量，Digital层的每个胶囊为16D向量，而采用的是全连接形式，因此需要一个胶囊对之间需要8x16的连接矩阵Weigth
        # # 且PrimaryCaps层共有1152个初级胶囊，DigitCaps层共有10个数字胶囊。full conneted则有1152x10对胶囊对（对于一幅图像而言）
        # w_initializer = np.random.normal(size=[1, 3, 1], scale=0.01)  # 通过训练获得
        # W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
        # # tf.tile( ）的核心作用是指定某一维度被复制扩展若干倍，最终的输出张量维度不变。
        # # 作用：当被“铺贴”的张量维度合适时，便可做矩阵（张量）运算，从而实现更高效的for循环。
        # #  [1, 1152, 8, 16] ---tile---> [batch_size, 1152, 8, 16]
        # W_Ij = tf.tile(W_Ij, [10, 1, 1]) # tile第二个参数分别指定各个维度扩展几倍
        #
        # # 计算 u_hat（Prediction Vector底层胶囊i到高层胶囊j的预测向量）  u_hat（u^）=u(也就是input)*W
        # # [8, 16].T * [8, 1] => [16, 1] => [batch_size, 1152, 16, 1]
        # u_hat = tf.matmul(W_Ij, input, transpose_a=True)   # transpose_a=True, 则a在进行乘法计算前进行转置。
        # assert u_hat.get_shape() == [15, 1152, 16, 1]
        #
        # # b_IJ shape=[1, 1152, 10, 1]
        # shape = b_IJ.get_shape().as_list()  # get_shape返回的是元组，需要通过as_list()的操作转换成list.
        # size_splits = [idx_j, 1, shape[2] - idx_j - 1]
        b_IJ = tf.zeros(shape=[FLAGS.batch_size, FLAGS.batch_size / 3], dtype=np.float32)
        #  -----------动态路由求耦合系数（Coupling Coefficent）c_ij-------------------------------------
        for r_iter in range(3):  # 迭代次数
            # [1, 1152, 10, 1] 在第二个 axis （数轴）上做归一化操作， 原因就是每一个 capsl 到所有 caps2 的概率总和为 l 。
            c_IJ = tf.nn.softmax(b_IJ, dim=1)  # DyRoting1、归一化c1,c2... =sotfmax(b1,b2...)   可见b和c是相关的，动态改变b也就改变了c
            # assert c_IJ.get_shape() == [1, 1152, 10, 1]

            # 在第三维使用c_I加权u_hat
            # 接着在第二维累加，得到[batch_size, 1, 16, 1]

            # 例如：在第二维度按[0,1,9]切b_IJ shape=[1, 1152, 10, 1] =>
            #                 b_Il[1,1152,0,1]
            #                 b_Ij[1,1152,1,1]
            #                 b_Ir[1,1152,9,1]
            # 为什么要切啊？？？？？ 后面又concat
            # b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis=2)
            # c_Il, c_Ij, b_Ir = tf.split(c_IJ, size_splits, axis=2)
            # assert c_Ij.get_shape() == [1, 1152, 1, 1]

            # s_j = tf.multiply(c_Ij, u_hat)   # 高层胶囊的加权输入    multiply求点乘
            # s_j = tf.reduce_sum(tf.matmul(input, c_IJ, transpose_a=True),   # u_hat (30,256) # DyRoting2、加权求和s = *u1 + c2*u2 + ...
            #                     axis=1, keep_dims=True)
            s_j = tf.matmul(input, c_IJ, transpose_a=True)

            # assert s_j.get_shape() == [cfg.batch_size, 1, 16, 1]

            # 第六行:
            # 使用上文提及的squash函数，得到：[batch_size, 1, 16, 1]
            v_j = squash(s_j)  # DyRoting3、 squashing压缩 即得到输出vj
            # assert s_j.get_shape() == [cfg.batch_size, 1, 16, 1]

            # 第7行:
            # 平铺v_j，由[batch_size ,1, 16, 1] 至[batch_size, 1152, 16, 1]
            # [16, 1].T x [16, 1] => [1, 1]
            # 接着在batch_size维度递归运算均值，得到 [1, 1152, 1, 1]
            # v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1])
            # u_produce_v = tf.matmul(input, v_j, transpose_a=True)    # DyRoting4、预测输出
            u_produce_v = tf.matmul(input, v_j)
            # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 1, 1]
            b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)  # DyRoting5、更新权值b b = v_j * u(输入的向量) + 之前的b
            # b_IJ = tf.concat([b_Il, b_Ij, b_Ir], axis=2)

        # return(v_j, b_IJ)  # v_j高层胶囊输出
        return tf.transpose(s_j)


def squash(vector):  # 参考压缩公式
    '''压缩函数:利用非线性挤压函数来完成这个归一化操作，它能保留向量的方向，
    同时把模长压缩至 l 以内。
    参数：
        vector：一个4维张量 [batch_size, num_caps, vec_len, 1],
    返回：
        一个和vector形状相同的4维张量，
        但第3维和第4维经过压缩
    '''
    vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))  # 一个标量
    scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
    vec_squashed = scalar_factor * tf.divide(vector, vec_abs)  # 对应元素相乘
    return (vec_squashed)


def npairSplit(embedding, Loss_type=FLAGS.LossType, size=1024):
    if Loss_type == 'NpairLoss':
        embedding_yp = tf.slice(input_=embedding, begin=[0, 0], size=[FLAGS.batch_size, size])
        embedding_yq = tf.slice(
            input_=embedding,
            begin=[FLAGS.batch_size, 0],
            size=[int(FLAGS.batch_size / 2 + np.square(FLAGS.batch_size / 2)), size])  # 生成的样本有8+8*8
        # size=[int(FLAGS.batch_size / 2 + FLAGS.batch_size / 2), size])
        return embedding_yp, embedding_yq
    elif Loss_type == 'easy_pos_hard_negLoss':
        embedding_yp = tf.slice(input_=embedding, begin=[0, 0], size=[FLAGS.batch_size, size])
        embedding_yq = tf.slice(
            input_=embedding,
            begin=[FLAGS.batch_size, 0],
            size=[int(FLAGS.batch_size / 2 + np.square(FLAGS.batch_size / 2)), size])  # 生成的样本有8+8*8
        # size=[int(FLAGS.batch_size / 2 + FLAGS.batch_size / 2), size])
        return embedding_yp, embedding_yq
    else:
        print("Not n-pair-loss")


def cross_entropy(embedding, label, size=1024):  # 原本1024
    with tf.variable_scope("Softmax_classifier"):
        W_fc = weight_variable([size, FLAGS.num_class], "softmax_w", wd=False)
        b_fc = bias_variable([FLAGS.num_class], "softmax_b")
    # Logits = tf.matmul(embedding, W_fc) + b_fc
    # Logits = tf.nn.leaky_relu(tf.matmul(embedding, W_fc) + b_fc)
    Logits = tf.nn.relu(tf.matmul(embedding, W_fc) + b_fc)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=Logits))
    return cross_entropy, W_fc, b_fc


# def cross_entropy(embedding, label, size=1024):  # 原本1024
#     with tf.variable_scope("Softmax_classifier"):
#         W_fc1 = weight_variable([size, size/2], "softmax_w1", wd=False)
#         b_fc1 = bias_variable([size/2], "softmax_b1")
#         W_fc2 = weight_variable([size/2, FLAGS.num_class], "softmax_w2", wd=False)
#         b_fc2 = bias_variable([FLAGS.num_class], "softmax_b2")
#     fc1 = tf.matmul(embedding, W_fc1) + b_fc1
#     Logits = tf.matmul(fc1, W_fc2) + b_fc2
#     cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=Logits))
#     return cross_entropy, W_fc1, b_fc1, W_fc2, b_fc2


def genSoftmax(embedding_anc, embedding_neg, W_fc, b_fc, label, Loss_type=FLAGS.LossType):
    if Loss_type == 'NpairLoss':
        label_split = tf.split(label, 2, axis=0)
        label_pos = tf.reshape(label_split[1], [int(FLAGS.batch_size / 2), 1])
        label_neg_tile = tf.tile(label_pos, [int(FLAGS.batch_size / 2), 1])

        pull_Logits = tf.matmul(embedding_neg, W_fc) + b_fc
        anc_Logits = tf.matmul(embedding_anc, W_fc) + b_fc
        label_neg_tile_2 = tf.reshape(label_neg_tile, [-1])
        label_anc_2 = tf.reshape(label_split[0], [-1])
        gen_cross_entropy = FLAGS.Softmax_factor * FLAGS._lambda * (
                tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_neg_tile_2, logits=pull_Logits))
                + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_anc_2, logits=anc_Logits))
        )
        return gen_cross_entropy
    elif Loss_type == 'easy_pos_hard_negLoss':
        label_split = tf.split(label, 2, axis=0)
        label_pos = tf.reshape(label_split[1], [int(FLAGS.batch_size / 2), 1])
        label_neg_tile = tf.tile(label_pos, [int(FLAGS.batch_size / 2), 1])

        pull_Logits = tf.matmul(embedding_neg, W_fc) + b_fc
        anc_Logits = tf.matmul(embedding_anc, W_fc) + b_fc
        label_neg_tile_2 = tf.reshape(label_neg_tile, [-1])
        label_anc_2 = tf.reshape(label_split[0], [-1])
        gen_cross_entropy = FLAGS.Softmax_factor * FLAGS._lambda * (
                tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_neg_tile_2, logits=pull_Logits))
                + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_anc_2, logits=anc_Logits))
        )
        return gen_cross_entropy
