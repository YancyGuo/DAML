from flags.FLAGS_HDML_triplet import *
from datasets import data_provider
from prostate_inference import inference
from lib import Loss_ops, HDML, nn_Ops, evaluation
from tqdm import tqdm
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# create a saver
# check system time
_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'/'

if FLAGS. SaveVal:
    nn_Ops.create_path(_time)
summary_writer = tf.summary.FileWriter(LOGDIR)


def train(stream_train, stream_test, stream_train_eval):
    """

    :type epoch_iterator: 训练集迭代对象
    """
    # 定义输入placeholder
    # placeholders Tensor("Placeholder:0", shape=(?, 227, 227, 3), dtype=float32)
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    # Tensor("Placeholder_1:0", shape=(?, 1), dtype=int32)
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    # Get the Label  （batch_size，1）-->（batch_size) 降维？
    label = tf.reduce_mean(label_raw, axis=1, keep_dims=False)

    # Training
    epoch_iterator = stream_train.get_epoch_iterator()

    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)
        if FLAGS.Apply_HDML:
            lr_gen = tf.placeholder(tf.float32)
            s_lr = tf.placeholder(tf.float32)

    if not FLAGS.Apply_HDML:
        # 定义前向传播过程
        embedding_z = inference(x_raw, is_Training)

        # 定义损失函数、学习率、训练过程
        # conventional Loss function 传统的损失函数
        with tf.name_scope('Loss'):
            # wdLoss = layers.apply_regularization(regularizer, weights_list=None)
            def exclude_batch_norm(name):
                """
                排除'Generator'、'Loss' scope中的'batch_normalization'节点
                :param name:
                :return:
                """
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            # 计算正则项：排除个别参数后，计算其余可训练参数的l2_loss后求和，并乘以权重衰减系数，作为wdLoss
            # 注：tf.nn.l2_loss：利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半
            wdLoss = FLAGS.Regular_factor * tf.add_n(  # FLAGS.Regular_factor权重衰减系数
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            )
            # wdLoss = FLAGS.Regular_factor * tf.add_n(  # FLAGS.Regular_factor权重衰减系数
            #     [tf.cond(tf.is_nan(tf.nn.l2_loss(v)),lambda: tf.constant(0.), lambda: tf.nn.l2_loss(v)) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            # )

            # For some kinds of Losses, the embedding should be l2 normed
            # embedding_l = embedding_z
            # 计算损失并加一个正则项
            # 正则化的基本思想是向损失函数添加一个惩罚项用于惩罚大的权重，隐式地减少自由参数的数量，
            # 所以可以达到弹性地适用不同数据量训练的要求而不产生过拟合的问题。
            # 正则化方法是将惩罚因子加入到各层的参数或激活函数中。
            # 其实现位置通常是在模型的optimization里，在计算损失函数时将该惩罚因子加进去。

            # 前半部分设置为交叉熵损失看看，然后不应用HDML
            J_m = Loss_ops.Loss(embedding_z, label, FLAGS.LossType) + wdLoss  # 加入正则项防止过拟合

            # 不应用HDML，创建训练节点，损失函数为传统的softmax或三元组损失等
            train_step = nn_Ops.training(loss=J_m, lr=lr)

            # learning rate
            _lr = FLAGS.init_learning_rate

            # collectors
            J_m_loss = nn_Ops.data_collector(tag='Jm', init=1e+6)
            wd_Loss = nn_Ops.data_collector(tag='weight_decay', init=1e+6)
            max_f1_score = 0
            step = 0

            # 循环训练并保存模型
            saver = tf.train.Saver()
            # initialise the session
            with tf.Session(config=config) as sess:
                # Initial all the variables with the sess
                sess.run(tf.global_variables_initializer())

                bp_epoch = FLAGS.init_batch_per_epoch  # 500
                with tqdm(total=FLAGS.max_steps) as pbar:
                    for batch in copy.copy(epoch_iterator):
                        # get images and labels from batch
                        x_batch_data, Label_raw = nn_Ops.batch_data(batch)
                        pbar.update(1)

                        train, J_m_var, wd_Loss_var, _embedding_z = sess.run([train_step, J_m, wdLoss, embedding_z],
                                                                             feed_dict={x_raw: x_batch_data,
                                                                                        label_raw: Label_raw,
                                                                                        is_Training: True,
                                                                                        # 喂入初始学习率,衰减在各个train op 里面设置
                                                                                        lr: _lr})
                        J_m_loss.update(var=J_m_var)
                        wd_Loss.update(var=wd_Loss_var)
                        step += 1

                        # evaluation
                        if step % bp_epoch == 0:
                            print('\n', 'only eval eval')
                            # nmi_tr, f1_tr, recalls_tr = evaluation.Evaluation(
                            # stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)
                            # nmi_te, f1_te, recalls_te = evaluation.Evaluation(
                            #     stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, FLAGS.batch_size, neighbours)
                            nmi_te, f1_te, recalls_te, recall, precision, f1_score = evaluation.Evaluation(
                                stream_test, stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training,
                                embedding_z, 2,
                                neighbours)

                            # Summary
                            eval_summary = tf.Summary()
                            # eval_summary.value.add(tag='train nmi', simple_value=nmi_tr)
                            # eval_summary.value.add(tag='train f1', simple_value=f1_tr)
                            # for i in range(0, np.shape(neighbours)[0]):
                            #     eval_summary.value.add(tag='Recall@%d train' % neighbours[i], simple_value=recalls_tr[i])
                            eval_summary.value.add(tag='test nmi', simple_value=nmi_te)
                            eval_summary.value.add(tag='test f1', simple_value=f1_te)
                            eval_summary.value.add(tag='recall', simple_value=recall)
                            eval_summary.value.add(tag='precision', simple_value=precision)
                            eval_summary.value.add(tag='f1_score', simple_value=f1_score)
                            for i in range(0, np.shape(neighbours)[0]):
                                eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te[i])
                            J_m_loss.write_to_tfboard(eval_summary)
                            wd_Loss.write_to_tfboard(eval_summary)
                            eval_summary.value.add(tag='learning_rate', simple_value=_lr)

                            summary_writer.add_summary(eval_summary, step)
                            print('Summary written')
                            if f1_score > max_f1_score:
                                max_f1_score = f1_score
                                print("Saved")
                                saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                            summary_writer.flush()

                            if step >= 7000:
                                bp_epoch = FLAGS.batch_per_epoch  # 测试频率设为这个
                            if step >= FLAGS.max_steps:  # 加上这个才能结束训练
                                os._exit()

    else:
        # 占位符
        with tf.name_scope('Javg'):
            Javg = tf.placeholder(tf.float32)
        with tf.name_scope('Jgen'):
            Jgen = tf.placeholder(tf.float32)
        # 定义前向传播过程
        embedding_z, embedding_y_origin, embedding_zq_anc, \
        embedding_zq_negtile, embedding_yp, embedding_yq = inference(x_raw, is_Training)
        # 定义损失函数、学习率、训练过程

        with tf.name_scope('Loss'):
            # 原始样本损失部分
            # wdLoss = layers.apply_regularization(regularizer, weights_list=None)
            def exclude_batch_norm(name):
                """
                排除'Generator'、'Loss' scope中的'batch_normalization'节点
                :param name:
                :return:
                """
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            # 计算正则项：排除个别参数后，计算其余可训练参数的l2_loss后求和，并乘以权重衰减系数，作为wdLoss
            # 注：tf.nn.l2_loss：利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半
            wdLoss = FLAGS.Regular_factor * tf.add_n(  # FLAGS.Regular_factor权重衰减系数
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            )
            # 前半部分设置为交叉熵损失看看，然后不应用HDML
            J_m = Loss_ops.Loss(embedding_z, label, FLAGS.LossType) + wdLoss  # 加入正则项防止过拟合

            # 22222生成部分损失
            # 训练度量学习的目标函数 论文公式11
            J_syn, reg_anchor, reg_positive, l2loss, pos_tile, anc, pos, label2, label_anc, label_pos, label_anc, similarity_matrix, labels_remapped, x_loss = Loss_ops.new_npair_loss(  # 合成样本的损失
                labels=label,  # (16,1)
                embedding_anchor=embedding_zq_anc,  # (8,128)
                embedding_positive=embedding_zq_negtile,  # (64,128)
                equal_shape=False, reg_lambda=FLAGS.loss_l2_reg)
            # J_syn = (1. - tf.exp(-FLAGS.beta / Jgen)) * J_m_generatesample
            # J_m = (tf.exp(-FLAGS.beta/Jgen))*J_m      # 原始样本的损失
            # J_metric = J_m + J_syn

            # J_metric = J_m_generatesample  # 分阶段训练不加上J_m试试

            # 生成器的目标函数 公式9
            cross_entropy, W_fc, b_fc = HDML.cross_entropy(embedding=embedding_y_origin, label=label)

            # 生成样本y切分成anchor和neg,其中anchor还是原来的，但参与了映射回y的过程
            embedding_yq_anc = tf.slice(
                input_=embedding_yq, begin=[0, 0], size=[int(FLAGS.batch_size / 2), 1024])
            embedding_yq_negtile = tf.slice(
                input_=embedding_yq, begin=[int(FLAGS.batch_size / 2), 0],
                size=[int(np.square(FLAGS.batch_size / 2)), 1024]   # (64, 1024)
            )

            # embedding_split = tf.split(embedding, 2, axis=0)  # （FLAGS.batch_size， 128）
            # embedding_y_origin_neg = embedding_split[1]
            # embedding_y_origin_neg = tf.reshape(tf.tile(embedding_y_origin_neg, [1, int(FLAGS.batch_size / 2)]),
            #                   [-1, 1024])

            # 原始样本重建损失
            J_recon = (1 - FLAGS._lambda) * tf.reduce_sum(tf.square(embedding_yp - embedding_y_origin))

            # J_soft = FLAGS._lambda * tf.reduce_sum(tf.square(embedding_yq_negtile - embedding_y_origin_neg))

            # 生成样本保持标签一致
            J_soft = HDML.genSoftmax(
                embedding_anc=embedding_yq_anc, embedding_neg=embedding_yq_negtile,
                W_fc=W_fc, b_fc=b_fc, label=label
            )
            J_gen = J_recon + J_soft

            train_step = nn_Ops.training(loss=J_m, lr=lr)  # 加
            c_train_step = nn_Ops.training(loss=J_syn, lr=lr, var_scope='Classifier')
            g_train_step = nn_Ops.training(loss=J_gen, lr=lr_gen, var_scope='Generator')  # 这两个学习率没有衰减？
            s_train_step = nn_Ops.training(loss=cross_entropy, lr=s_lr, var_scope='Softmax_classifier')

            # learning rate
            _lr = FLAGS.init_learning_rate
            _lr_gen = FLAGS.lr_gen
            _s_lr = FLAGS.s_lr

            # Training
            epoch_iterator = stream_train.get_epoch_iterator()

            # collectors
            J_m_loss = nn_Ops.data_collector(tag='Jm', init=1e+6)
            J_syn_loss = nn_Ops.data_collector(tag='J_syn', init=1e+1)
            J_metric_loss = nn_Ops.data_collector(tag='J_metric', init=1e+1)
            J_soft_loss = nn_Ops.data_collector(tag='J_soft', init=1e+1)
            J_recon_loss = nn_Ops.data_collector(tag='J_recon', init=1e+1)
            J_gen_loss = nn_Ops.data_collector(tag='J_gen', init=1e+1)
            cross_entropy_loss = nn_Ops.data_collector(tag='cross_entropy', init=1e+6)
            wd_Loss = nn_Ops.data_collector(tag='weight_decay', init=1e+6)
            max_nmi = 0
            max_f1_score = 0
            step = 0

            # -----------------循环训练并保存模型------------
            # 初始化tensorflow持久化类
            saver = tf.train.Saver()
            with tf.Session(config=config) as sess:
                # Initial all the variables with the sess
                sess.run(tf.global_variables_initializer())

                bp_epoch = FLAGS.init_batch_per_epoch  # 500
                with tqdm(total=FLAGS.max_steps) as pbar:
                    for batch in copy.copy(epoch_iterator):
                        # get images and labels from batch
                        x_batch_data, Label_raw = nn_Ops.batch_data(batch)
                        pbar.update(1)

                        if step <= 8000:  # ----------分段训练------------------
                            # s_train, wd_Loss_var, J_m_var, cross_en_var, _label_raw = sess.run(
                            #     [s_train_step, wdLoss, J_m,
                            #    cross_entropy, label_raw],
                            #     feed_dict={x_raw: x_batch_data,
                            #                label_raw: Label_raw,
                            #                is_Training: True})
                            #
                            # wd_Loss.update(var=wd_Loss_var)
                            # J_m_loss.update(var=J_m_var)
                            # cross_entropy_loss.update(cross_en_var)

                            train, wd_Loss_var, J_m_var, cross_en_var, _label_raw = sess.run(
                                [train_step, wdLoss, J_m, cross_entropy, label_raw],
                                feed_dict={x_raw: x_batch_data,
                                           label_raw: Label_raw,
                                           is_Training: True,
                                           # 喂入初始学习率,衰减在各个train op 里面设置
                                           lr: _lr, lr_gen: _lr_gen, s_lr: _s_lr,
                                           Javg: J_m_loss.read(), Jgen: J_gen_loss.read()})

                            wd_Loss.update(var=wd_Loss_var)
                            J_m_loss.update(var=J_m_var)
                            cross_entropy_loss.update(cross_en_var)

                            # elif step > 2000 & step <= 4000:
                        else:
                            c_train, g_train, s_train, wd_Loss_var, J_m_var, \
                            J_syn_var, J_recon_var, J_soft_var, J_gen_var, cross_en_var, \
                            _pos_tile, _reg_anchor, _reg_positive, _l2loss, _pos_tile, _anc, _pos, _label2, \
                            _label_anc, _label_pos, _label_anc, _similarity_matrix, _labels_remapped, _x_loss = sess.run(
                                [c_train_step, g_train_step, s_train_step, wdLoss,
                                 J_m, J_syn, J_recon, J_soft, J_gen, cross_entropy,
                                 pos_tile, reg_anchor, reg_positive,
                                 l2loss, pos_tile, anc, pos, label2, label_anc, label_pos,
                                 label_anc, similarity_matrix, labels_remapped, x_loss],
                                feed_dict={x_raw: x_batch_data,
                                           label_raw: Label_raw,
                                           is_Training: True,
                                           # 喂入初始学习率,衰减在各个train op 里面设置
                                           lr: _lr, lr_gen: _lr_gen, s_lr: _s_lr,
                                           Javg: J_m_loss.read(), Jgen: J_gen_loss.read()})

                            wd_Loss.update(var=wd_Loss_var)
                            # J_metric_loss.update(var=J_metric_var)
                            J_m_loss.update(var=J_m_var)
                            J_syn_loss.update(var=J_syn_var)
                            J_recon_loss.update(var=J_recon_var)
                            J_soft_loss.update(var=J_soft_var)
                            J_gen_loss.update(var=J_gen_var)
                            cross_entropy_loss.update(cross_en_var)
                        step += 1

                        # evaluation
                        if step % bp_epoch == 0:
                            print('\n', 'only eval eval')
                            # nmi_tr, f1_tr, recalls_tr = evaluation.Evaluation(
                            # stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)
                            # nmi_te, f1_te, recalls_te = evaluation.Evaluation(
                            #     stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, FLAGS.batch_size, neighbours)
                            # nmi_te, f1_te, recalls_te, recall, precision, f1_score = evaluation.Evaluation(
                            #     stream_train_eval, stream_train, image_mean, sess, x_raw, label_raw, is_Training,
                            #     embedding_z, 2,
                            #     neighbours)  # 验证集

                            nmi_te, f1_te, recalls_te, recall, precision, f1_score = evaluation.Evaluation(
                                stream_test, stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training,
                                embedding_z, 2,
                                neighbours)

                            # Summary
                            eval_summary = tf.Summary()
                            # eval_summary.value.add(tag='train nmi', simple_value=nmi_tr)
                            # eval_summary.value.add(tag='train f1', simple_value=f1_tr)
                            # for i in range(0, np.shape(neighbours)[0]):
                            #     eval_summary.value.add(tag='Recall@%d train' % neighbours[i], simple_value=recalls_tr[i])
                            eval_summary.value.add(tag='test nmi', simple_value=nmi_te)
                            eval_summary.value.add(tag='test f1', simple_value=f1_te)
                            eval_summary.value.add(tag='recall', simple_value=recall)
                            eval_summary.value.add(tag='precision', simple_value=precision)
                            eval_summary.value.add(tag='f1_score', simple_value=f1_score)
                            for i in range(0, np.shape(neighbours)[0]):
                                eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te[i])
                            J_m_loss.write_to_tfboard(eval_summary)
                            wd_Loss.write_to_tfboard(eval_summary)
                            eval_summary.value.add(tag='learning_rate', simple_value=_lr)
                            if FLAGS.Apply_HDML:
                                J_syn_loss.write_to_tfboard(eval_summary)
                                J_metric_loss.write_to_tfboard(eval_summary)
                                J_soft_loss.write_to_tfboard(eval_summary)
                                J_recon_loss.write_to_tfboard(eval_summary)
                                J_gen_loss.write_to_tfboard(eval_summary)
                                cross_entropy_loss.write_to_tfboard(eval_summary)
                            summary_writer.add_summary(eval_summary, step)
                            print('Summary written')
                            if f1_score > max_f1_score:
                                max_f1_score = f1_score
                                print("Saved")
                                saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                            summary_writer.flush()

                        if step >= 7000:
                            bp_epoch = FLAGS.batch_per_epoch  # 测试频率设为这个
                        if step >= FLAGS.max_steps:  # 加上这个才能结束训练
                            os._exit()



# def main(argv=None):
#     # Create the stream of datas from dataset 创建数据流
#     streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, method, crop_size=FLAGS.default_image_size)
#     stream_train, stream_train_eval, stream_test = streams
#     train(stream_train, stream_test, stream_train_eval)

if __name__ == '__main__':
    streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, method=FLAGS.method, crop_size=FLAGS.default_image_size)
    stream_train, stream_train_eval, stream_test = streams
    train(stream_train, stream_test, stream_train_eval)