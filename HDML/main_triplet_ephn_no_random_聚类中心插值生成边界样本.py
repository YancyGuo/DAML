from datasets import data_provider
# import datasets.data_provider as data_provider
from lib import GoogleNet_Model, Loss_ops, nn_Ops, HDML_centroid_interpolation, evaluation
import copy
from tqdm import tqdm
from tensorflow.contrib import layers
from flags.FLAGS_HDML_triplet import *
# import faiss
# from tensorflow.python import debug as tf_debug
# from lib.utils import cosine
# from lib.utils import CosineEmbeddingLoss

# import clustering

# Using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create the stream of datas from dataset
streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, FLAGS.method, crop_size=FLAGS.default_image_size)
streams_no_aug, stream_train, stream_train_eval, stream_test = streams

regularizer = layers.l2_regularizer(FLAGS.Regular_factor)
# create a saver
# check system time
_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'/'

if FLAGS. SaveVal:
    nn_Ops.create_path(_time)
summary_writer = tf.summary.FileWriter(LOGDIR)



def main(_):
    # if not FLAGS.LossType == 'Triplet':
    #     print("LossType triplet loss is required")
    #     return 0

    # placeholders
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    x_flat = tf.reshape(x_raw, shape=[-1, FLAGS.default_image_size * FLAGS.default_image_size * 3])
    # batch_size的一半9
    # idx_0 = tf.random_shuffle(tf.range(9))
    #     # idx_1 = tf.random_shuffle(tf.range(9, tf.shape(x_flat)[0]))
    #     # x_flat_permute_0 = tf.gather(x_flat, idx_0)
    #     # x_flat_permute_1 = tf.gather(x_flat, idx_1)
    # x_flat_permute = tf.concat([x_flat_permute_0, x_flat_permute_1], axis=0)
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])

    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)

    with tf.variable_scope('Classifier'):
        google_net_model = GoogleNet_Model.GoogleNet_Model()
        embedding = google_net_model.forward(x_raw)

        # Batch Normalization layer 1
        embedding = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1')
        # FC layer 1
        embedding_z = nn_Ops.fc_block(
            embedding, in_d=1024, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)

        # embedding_z = tf.Print(embedding_z, [embedding_z], "骨干网络的嵌入", summarize=1000)


        # Embedding Visualization
        # assignment, embedding_var = Embedding_Visualization.embedding_assign(
        #     batch_size=256, embedding=embedding_z,
        #     embedding_size=FLAGS.embedding_size, name='Embedding_of_fc1')

        # conventional Loss function
        # with tf.name_scope('Loss'):
            # # wdLoss = layers.apply_regularization(regularizer, weights_list=None)
            # def exclude_batch_norm(name):
            #     return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name
            #
            # wdLoss = FLAGS.Regular_factor * tf.add_n(
            #     [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            # )
            # Get the Label
        label = tf.reduce_mean(label_raw, axis=1, keep_dims=False)
            # For some kinds of Losses, the embedding should be l2 normed
        embedding_z = tf.nn.l2_normalize(embedding_z, dim=1)

            # prior_jm = tf.Print(prior_jm, [prior_jm], "prior_jm", summarize=100)
            # ori_loss, I_pos, positive_emb, I_neg, negative_emb, positive_label, negative_label = Loss_ops.Loss(embedding_z, label, 0.1, _lossType=FLAGS.LossType)
            #
            # # J_m = ori_loss + wdLoss
            # J_m = ori_loss

    # if FLAGS.Apply_HDML:

        # # # 应该是同类进行permute
        # # # embedding_y_origin_permute = tf.random_shuffle(embedding)
        # emb0, emb1 = tf.split(embedding, 2, axis=0)
        # #
        # emb0_permute = tf.gather(emb0, tf.random_shuffle(tf.range(tf.shape(emb0)[0])))
        # emb1_permute = tf.gather(emb1, tf.random_shuffle(tf.range(tf.shape(emb1)[0])))
        # #
        # # # emb1_permute = tf.random_shuffle(emb1)
        # embedding_y_origin_permute = tf.concat([emb0_permute, emb1_permute], axis=0)

    # if HNG is applied
    if FLAGS.Apply_HDML:
        with tf.name_scope('current_epoch'):
            current_epoch = tf.placeholder(tf.float32)


        # 在嵌入空间合成的 hardness-aware tuple [anc, pos, neg]  (主要是合成了neg)
        # nan是因为生成的样本使reuse的y-->z的全连接层出现问题了
        # embedding_z_recombine = tf.concat([embedding_z, positive_emb, negative_emb], axis=0)
        # concat_emb = tf.concat(embeddings, inner_pts_1, inner_pts_2, dim=0)
        concat_embeddings, concat_labels = \
            HDML_centroid_interpolation.Pulling\
                (FLAGS.LossType, embedding_z, label, Lambda_0=1.0, T=FLAGS.max_steps, t=current_epoch)  # 2组样本  256维  原来的+插值点
        # dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        # 原来的和合成的tuple合并
        # embedding_z_concate = tf.concat([embedding_z, concat_embeddings], axis=0)
        # embedding_z_concate_labels = tf.concat([label, concat_labels], axis=0)

        # Generator
        # 将原样本与合成样本都映射回feature space
        # z--->y’,z^--->y~ ,maps the augmented embeding of a tuple back to the feature space.
        # maps not only the synthetic negative sample but other unaltered samples in one tuple.

        # # 将原样本映射回去feature space ,用于重建损失
        # with tf.variable_scope('Generator'):
        #     # generator fc3
        #     embedding_y = nn_Ops.fc_block(
        #         embedding_z, in_d=FLAGS.embedding_size, out_d=512,
        #         name='generator1', is_bn=True, is_relu=True, is_Training=is_Training
        #     )
        #
        #     # generator fc4???
        #     embedding_y = nn_Ops.fc_block(
        #         embedding_y, in_d=512, out_d=1024,
        #         name='generator2', is_bn=False, is_relu=False, is_Training=is_Training
        #     )
        #
        #     # 重建图像
        #     embedding_yp_img = nn_Ops.fc_block(
        #         embedding_y, in_d=1024, out_d=FLAGS.default_image_size * FLAGS.default_image_size * 3,
        #         name='generator_img', is_bn=False, is_relu=False, is_Training=is_Training
        #     )
        #
        #     embedding_yp_img = tf.nn.sigmoid(embedding_yp_img)
        #     embedding_yp_img = tf.reshape(embedding_yp_img,
        #                                   shape=[-1, FLAGS.default_image_size, FLAGS.default_image_size, 3])
            # unreshape = tf.reshape(embedding_yp_img, [-1, FLAGS.default_image_size * FLAGS.default_image_size * 3])

            #  embedding_yp是原样本映射回去的， 而embedding_yq是合成样本映射的
            # embedding_yp, embedding_yq = tf.split(embedding_y_concate, 2, axis=0)
            # embedding_yp_orisample = tf.gather(embedding_y_concate, tf.range(FLAGS.batch_size))
            # embedding_yq = tf.gather(embedding_y_concate, tf.range(FLAGS.batch_size, FLAGS.batch_size + tf.shape(concat_embeddings)[0])) # 合成点
            # embedding_yp_orisample, _, _ = tf.split(embedding_yp, 3, axis=0)
            # embedding_yq_labels = tf.concat([label, positive_label, negative_label], axis=0)

        # # 用合成的样本训练feature space到embedding space的encoder全连接层？？
        # with tf.variable_scope('Classifier'):
        #     # （只有这两层是用生成样本训练了，感觉并没有训练到提取特征的Backbone）
        #     embedding_z_quta = nn_Ops.bn_block(
        #         embedding_yq, normal=FLAGS.normalize, is_Training=is_Training, name='BN1', reuse=True)
        #
        #     embedding_z_quta = nn_Ops.fc_block(  # shape: (30, 256)有一个256是nan
        #         embedding_z_quta, in_d=1024, out_d=FLAGS.embedding_size,
        #         name='fc1', is_bn=False, is_relu=False, reuse=True, is_Training=is_Training
        #     )

        with tf.name_scope('Loss'):
            # J_syn 用的是生成样本，J_m用的是原始样本
            # 这里的loss函数和前面J_m的是一样的
            # embedding_z_concate = tf.nn.l2_normalize(concat_embeddings, dim=1)

            loss = Loss_ops.Loss(concat_embeddings, concat_labels, 0.1, _lossType='ephn_label_preserving_loss')  # ephn_label_preserving_loss



            # J_metric = J_m + J_syn  # 总的度量学习损失

            # 重建损失Jrecon = ||y − y′||2 2 to restrict the encoder & decoder to
            # map each point close to itself.
            # 因为y----encoder--->z----decoder(生成器)--->y'，所以要使映射回feature space的y'接近它本身
            # # 映射后的原样本与原样本    原来是reduce_sum
            # noise = tf.reduce_mean(tf.square(embedding_y_origin_permute - embedding))
            # # J_recon = tf.reduce_sum(tf.square(embedding_yp_orisample - embedding_y_origin_permute))  # 1024维
            # J_recon = tf.reduce_sum(tf.square(embedding_y - embedding)) + noise  # 1024维
            # # J_recon = (1 - FLAGS._lambda) * tf.reduce_sum(tf.abs(embedding_yp - embedding_y_origin))

            # 图像重建
            # noise = tf.reduce_mean(tf.squared_difference(unreshape, x_flat_permute))
            # J_recon = tf.reduce_mean(tf.squared_difference(unreshape, x_flat))

            # J_recon = 10000 * tf.reduce_mean(1 - cosine(embedding_yp, embedding_y_origin)) # 改为cosin相似度
            # J_recon = 80 * tf.reduce_sum(1 - cosine(embedding_yp, embedding_y_origin))
            # J_recon = 200 * CosineEmbeddingLoss(margin=0.5)(embedding_yp, embedding_y_origin, embedding_y_origin_label)

            # J_recon = tf.cond(tf.is_nan(J_recon), lambda: tf.constant(0.), lambda: J_recon)

            # 使用原始的特征y训练全连接层，然后用训练好的参数用来保证生成样本标签一致。
            # we simultaneously learn a fully connected layer with the softmax loss on y,
            # where the gradients only update the parameters in this layer.
            # We employ the learned softmax layer to compute the softmax loss
            # jsoft(y~, l) between the synthetic hardness-aware negative y~
            # and the original label l.
            # cross_entropy, W_fc1, b_fc1, W_fc2, b_fc2 = HDML.cross_entropy(embedding=embedding_y_origin, label=label)
            # q_fc1 = tf.matmul(embedding_yq, W_fc1) + b_fc1
            # Logits_q = tf.matmul(q_fc1, W_fc2) + b_fc2
            # cross_entropy, W_fc, b_fc = HDML.cross_entropy(embedding=embedding_y_origin, label=label)
            # cross_entropy, W_fc, b_fc, logits = HDML.cross_entropy(embedding=embedding_y_origin, label=label)
            # Logits_q = tf.matmul(embedding_yq, W_fc) + b_fc

            # cross_entropy, W_fc, b_fc = HDML_origin.cross_entropy(embedding=embedding_y_origin, label=embedding_y_origin_label)
            # cross_entropy = tf.cond(tf.is_nan(cross_entropy), lambda: tf.constant(0.), lambda: cross_entropy)
            # Logits_q = tf.nn.leaky_relu(tf.matmul(embedding_yq, W_fc) + b_fc)
            # Logits_q = tf.nn.relu(tf.matmul(embedding_yq, W_fc) + b_fc)

            # J_soft = FLAGS.Softmax_factor * FLAGS._lambda * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=embedding_yq_labels, logits=Logits_q))
            # J_soft = FLAGS._lambda * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=embedding_yq_labels, logits=Logits_q))

            # J_soft = tf.cond(tf.is_nan(J_soft), lambda: tf.constant(0.), lambda: J_soft)
            # J_gen = 10 * (J_recon + J_soft)
            # J_gen = J_recon + J_soft
            # J_gen = J_recon

    if FLAGS.Apply_HDML:
        # train_step = nn_Ops.training(loss=J_m, lr=lr)
        c_train_step = nn_Ops.training(loss=loss, lr=lr, var_scope='Classifier')
        # g_train_step = nn_Ops.training(loss=J_gen, lr=FLAGS.lr_gen, var_scope='Generator')
        # s_train_step = nn_Ops.training(loss=cross_entropy, lr=FLAGS.s_lr, var_scope='Softmax_classifier')
    else:
        train_step = nn_Ops.training(loss=J_m, lr=lr)

    # initialise the session
    with tf.Session(config=config) as sess:
        # Initial all the variables with the sess

        sess.run(tf.global_variables_initializer())
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline", dump_root="D:/model/debug")
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        saver = tf.train.Saver()

        # learning rate
        _lr = FLAGS.init_learning_rate

        # Restore a checkpoint
        if FLAGS.load_formalVal:
            saver.restore(sess, FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+FLAGS.formerTimer)
        
        # Training
        epoch_iterator = streams_no_aug.get_epoch_iterator()  # 不应用增强会出现nan

        # collectors
        J_m_loss = nn_Ops.data_collector(tag='Jm', init=1e+6)  # 空
        J_syn_loss = nn_Ops.data_collector(tag='J_syn', init=1e+6)
        J_metric_loss = nn_Ops.data_collector(tag='J_metric', init=1e+6)  # 空
        J_soft_loss = nn_Ops.data_collector(tag='J_soft', init=1e+6)
        J_recon_loss = nn_Ops.data_collector(tag='J_recon', init=1e+6)
        J_gen_loss = nn_Ops.data_collector(tag='J_gen', init=1e+6)
        cross_entropy_loss = nn_Ops.data_collector(tag='cross_entropy', init=1e+6)
        wd_Loss = nn_Ops.data_collector(tag='weight_decay', init=1e+6)

        max_nmi = 0
        max_f1_score = 0
        step = 0

        r_step = []
        r_mask = []
        r_neg = []
        r_neg2 = []

        bp_epoch = FLAGS.init_batch_per_epoch
        with tqdm(total=FLAGS.max_steps) as pbar:
            for batch in copy.copy(epoch_iterator):
                # get images and labels from batch
                x_batch_data, Label_raw = nn_Ops.batch_data(batch)
                pbar.update(1)

                if not FLAGS.Apply_HDML:
                    # train, J_m_var, wd_Loss_var = sess.run([train_step, J_m, wd_Loss],
                    #                                        feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                    #                                                   is_Training: True, lr: _lr})
                    train, J_m_var, = sess.run([train_step, loss],
                                                           feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                                                                      is_Training: True, lr: _lr, current_epoch:step})
                    J_m_loss.update(var=J_m_var)
                    # wd_Loss.update(var=wd_Loss_var)

                    # pbar.set_description("Jm: %f, wd_Loss_var: %f" % (J_m_var, wd_Loss_var))
                    pbar.set_description("Jm: %f" % (J_m_var))


                else:
                    # summaries = tf.summary.merge_all()
                    # c_train, g_train, s_train, wd_Loss_var, J_metric_var, J_m_var, \
                    #         J_syn_var, J_recon_var, J_soft_var, J_gen_var, cross_en_var, _label_raw, summ, neg2_, neg_, neg_mask_ = sess.run(
                    #         [c_train_step, g_train_step, s_train_step, wdLoss,
                    #          J_metric, J_m, J_syn, J_recon, J_soft, J_gen, cross_entropy, label_raw, summaries, neg2, neg, neg_mask],
                    #         feed_dict={x_raw: x_batch_d
                    #         ata,
                    #                    label_raw: Label_raw,
                    #                    is_Training: True, lr: _lr, Javg: J_m_loss.read(), Jgen: J_gen_loss.read()})

                    c_train, J_metric_var, \
                     _label_raw,\
                         = sess.run(
                        [c_train_step,
                         loss, label_raw,
                        ],
                        feed_dict={x_raw: x_batch_data,
                                   label_raw: Label_raw,
                                   is_Training: True, lr: _lr, current_epoch:step})

                    pbar.set_description("J_metric_var: %f"%
                                         (J_metric_var,  ))

                    # wd_Loss.update(var=wd_Loss_var)
                    J_metric_loss.update(var=J_metric_var)


                step += 1
                # print('learning rate %f' % _lr)

                # evaluation
                if step % bp_epoch == 0:
                    print('only eval eval')
                    nmi_te, f1_te, recalls_te, recall, precision, f1_score, k_max_f1score = evaluation.Evaluation(
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
                    # wd_Loss.write_to_tfboard(eval_summary)
                    eval_summary.value.add(tag='learning_rate', simple_value=_lr)
                    if FLAGS.Apply_HDML:
                        J_syn_loss.write_to_tfboard(eval_summary)
                        J_metric_loss.write_to_tfboard(eval_summary)
                        J_soft_loss.write_to_tfboard(eval_summary)
                        J_recon_loss.write_to_tfboard(eval_summary)
                        J_gen_loss.write_to_tfboard(eval_summary)
                        cross_entropy_loss.write_to_tfboard(eval_summary)
                        # prior_jm_loss.write_to_tfboard(eval_summary)  # 33
                    summary_writer.add_summary(eval_summary, step)
                    print('Summary written')
                    if k_max_f1score > max_f1_score:
                        max_f1_score = k_max_f1score
                        print("Saved with max_f1_score:", max_f1_score)
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                    summary_writer.flush()
                    if step in [5632, 6848]:  # 原来的
                    # if step in [1500, 6848]:
                        _lr = _lr * 0.5

                    if step >= 5000:
                        bp_epoch = FLAGS.batch_per_epoch
                    if step >= FLAGS.max_steps:
                        os._exit()

if __name__ == '__main__':
    tf.app.run()
