from datasets import data_provider
# import datasets.data_provider as data_provider
from lib import GoogleNet_Model, Loss_ops, nn_Ops, Embedding_Visualization, HDML_n_pnt, evaluation
import copy
from tqdm import tqdm
from tensorflow.contrib import layers
from flags.FLAGS import *

#  调整了一些错误：原样本和合成样本没有分好放进去训练合成损失 。permute只作为重建损失的噪声

# Using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create the stream of datas from dataset
streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, FLAGS.method, crop_size=FLAGS.default_image_size)
streams_no_aug, stream_train, stream_train_eval, stream_test = streams    # stream_train_eval是查询集
trainset_num = streams_no_aug.dataset.num_examples
decay_step = trainset_num // FLAGS.batch_size  # 一个时代的步数*10

regularizer = layers.l2_regularizer(FLAGS.Regular_factor)
# create a saver
# check system time
_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
dataset_splite_num = FLAGS.path.split('/')[-1].split("_")[-1].split(".")[0]  # 不需要标签一致插值重建permute_no_noise
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'_'+'emb32'+dataset_splite_num+'/'

if FLAGS. SaveVal:
    nn_Ops.create_path(_time)
summary_writer = tf.summary.FileWriter(LOGDIR)


def main(_):
    if not FLAGS.LossType == 'easy_pos_hard_negLoss':
        print("LossType easy_pos_hard_negLoss loss is required")
        return 0
    if not FLAGS.Apply_HDML:
        print("要用HDML")
        return 0

    # placeholders
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    # decay_steps = tf.placeholder(tf.int32)  # 衰减周期
    global_step1 = tf.Variable(0, trainable=False)  # 把他传到优化器里面进行更新
    global_step2 = tf.Variable(0, trainable=False)  # 把他传到优化器里面进行更新

    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    # with tf.name_scope('learning_rate'):
    #     starter_learning_rate = tf.placeholder(tf.float32)

    with tf.variable_scope('Classifier'):
        google_net_model = GoogleNet_Model.GoogleNet_Model()
        embedding = google_net_model.forward(x_raw)
        if FLAGS.Apply_HDML:
            embedding_y_origin = embedding

        # Batch Normalization layer 1
        embedding = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1')
        # FC layer 1
        embedding_z = nn_Ops.fc_block(
            embedding, in_d=1024, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)

        # embedding_z = tf.Print(embedding_z, [embedding_z], "骨干网络的嵌入", summarize=1000)


        # Embedding Visualization
        assignment, embedding_var = Embedding_Visualization.embedding_assign(
            batch_size=256, embedding=embedding_z,
            embedding_size=FLAGS.embedding_size, name='Embedding_of_fc1')

        # conventional Loss function
        with tf.name_scope('Loss'):
            # wdLoss = layers.apply_regularization(regularizer, weights_list=None)
            def exclude_batch_norm(name):
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            wdLoss = FLAGS.Regular_factor * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            )
            # Get the Label
            label = tf.reduce_mean(label_raw, axis=1, keep_dims=False)
            # For some kinds of Losses, the embedding should be l2 normed
            embedding_z = tf.nn.l2_normalize(embedding_z, dim=1)

            # prior_jm = tf.Print(prior_jm, [prior_jm], "prior_jm", summarize=100)
            ori_loss, I_pos, positive_emb, I_neg, negative_emb, positive_label, negative_label = Loss_ops.Loss(embedding_z, label, 0.1, _lossType=FLAGS.LossType)

            J_m = ori_loss + 0.01*wdLoss
            # J_m = ori_loss

    if FLAGS.Apply_HDML:
        positive_origin = tf.gather(embedding, I_pos)
        negative_origin = tf.gather(embedding, I_neg)

        # 应该是同类进行permute
        # embedding_y_origin_permute = tf.random_shuffle(embedding)

        # embedding_y_origin = tf.concat([embedding, positive_origin, negative_origin], axis=0)
        # embedding_y_origin_label = tf.concat([label, positive_label, negative_label], axis=0)
        # label = tf.Print(label, [label], "label", summarize=100)
        # positive_label = tf.Print(positive_label, [positive_label], "positive_label", summarize=100)
        # negative_label = tf.Print(negative_label, [negative_label], "negative_label", summarize=100)

    # if HNG is applied
    if FLAGS.Apply_HDML:
        with tf.name_scope('Javg'):
            Javg = tf.placeholder(tf.float32)
        with tf.name_scope('Jgen'):
            Jgen = tf.placeholder(tf.float32)

        # 在嵌入空间合成的 hardness-aware tuple [anc, pos, neg]  (主要是合成了neg)
        # nan是因为生成的样本使reuse的y-->z的全连接层出现问题了
        # embedding_z_recombine = tf.concat([embedding_z, positive_emb, negative_emb], axis=0)
        # concat_emb = tf.concat(embeddings, inner_pts_1, inner_pts_2, dim=0)

            # # 这是传入原样本和正负例的时候，记得注释掉！
            # sample_split = tf.split(embedding, 3, axis=0)
            # embedding = sample_split[0]  # （18，256）
            # embedding = tf.nn.l2_normalize(embedding, dim=1)
            #
            # # 只传入原样本时，上面的注释掉
            # samples = tf.split(embedding, 2, axis=0)  # （9，256）
            # class0_emb = samples[0]
            # class1_emb = samples[1]!!!!!!!!!!!
        concat_embeddings, concat_labels = HDML_n_pnt.Pulling(FLAGS.LossType, embedding_z, label)  # 4组样本  256维 已经包括原样本
        # dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        # 原来的和合成的tuple合并
        # embedding_z_concate = tf.concat([embedding_z, concat_embeddings], axis=0)


        # Generator
        # 将原样本与合成样本都映射回feature space
        # z--->y’,z^--->y~ ,maps the augmented embeding of a tuple back to the feature space.
        # maps not only the synthetic negative sample but other unaltered samples in one tuple.

        # generate之前需不需要归一化一下？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        with tf.variable_scope('Generator'):
            # generator fc3
            embedding_y_concate = nn_Ops.fc_block(
                concat_embeddings, in_d=FLAGS.embedding_size, out_d=512,
                name='generator1', is_bn=True, is_relu=True, is_Training=is_Training
            )

            # generator fc4???
            embedding_y_concate = nn_Ops.fc_block(
                embedding_y_concate, in_d=512, out_d=1024,
                name='generator2', is_bn=False, is_relu=False, is_Training=is_Training
            )

            #  embedding_yp是原样本映射回去的， 而embedding_yq是合成样本映射的
            # embedding_yp, embedding_yq = tf.split(embedding_y_concate, 2, axis=0)

            embedding_yq = tf.gather(embedding_y_concate, tf.range(FLAGS.batch_size, tf.shape(concat_embeddings)[0]))  # 合成

            embedding_yq_label = tf.gather(concat_labels, tf.range(FLAGS.batch_size, tf.shape(concat_labels)[0]))  # 合成
            # embedding_yp_orisample, _, _ = tf.split(embedding_yp, 3, axis=0)
            # embedding_yq_labels = tf.concat([label, positive_label, negative_label], axis=0)

        # 用合成的样本训练feature space到embedding space的encoder全连接层？？
        with tf.variable_scope('Classifier'):
            # （只有这两层是用生成样本训练了，感觉并没有训练到提取特征的Backbone）
            embedding_z_quta = nn_Ops.bn_block(
                embedding_yq, normal=FLAGS.normalize, is_Training=is_Training, name='BN1', reuse=True)

            embedding_z_quta = nn_Ops.fc_block(  # shape: (30, 256)有一个256是nan
                embedding_z_quta, in_d=1024, out_d=FLAGS.embedding_size,
                name='fc1', is_bn=False, is_relu=False, reuse=True, is_Training=is_Training
            )

        with tf.name_scope('Loss'):
            # J_syn 用的是生成样本，J_m用的是原始样本
            # 这里的loss函数和前面J_m的是一样的
            embedding_z_quta = tf.nn.l2_normalize(embedding_z_quta, dim=1)
            # positive 只用原样本
            # syn_sample_loss_loss = Loss_ops.Loss(embedding_z_quta, embedding_yq_label, 0.1, _lossType='hard_neg_mining_epnh_synstage_loss')
            # positive也在合成和原样本中找
            syn_sample_loss_loss, _, _, _, _, _, _ = Loss_ops.Loss(embedding_z_quta, embedding_yq_label, 0.1, _lossType=FLAGS.LossType)
            syn_factor = tf.exp(-FLAGS.beta/Jgen)
            # Javg = tf.clip_by_value(Jm, 1e-4, tf.reduce_max(Jm))
            J_syn = (1 - syn_factor) * syn_sample_loss_loss  # 原
            # J_syn = syn_sample_loss_loss
            J_m = syn_factor*J_m
            J_metric = J_m + J_syn  # 总的度量学习损失

            # 重建损失Jrecon = ||y − y′||2 2 to restrict the encoder & decoder to
            # map each point close to itself.

            # 因为y----encoder--->z----decoder(生成器)--->y'，所以要使映射回feature space的y'接近它本身
            # 映射后的原样本与原样本    原来是reduce_sum
            embedding_yp_orisample = tf.gather(embedding_y_concate, tf.range(FLAGS.batch_size))   # 生成器映射回来的原样本
            emb0, emb1 = tf.split(embedding, 2, axis=0)  # embedding是batch norm后的
            emb0_permute = tf.gather(emb0, tf.random_shuffle(tf.range(tf.shape(emb0)[0])))
            emb1_permute = tf.gather(emb1, tf.random_shuffle(tf.range(tf.shape(emb1)[0])))
            embedding_y_origin_permute = tf.concat([emb0_permute, emb1_permute], axis=0)

            noise = 10* tf.reduce_mean(tf.square(embedding_yp_orisample - embedding_y_origin_permute))  # 原
            J_recon = tf.reduce_sum(tf.square(embedding_yp_orisample - embedding)) + noise  # 1024维 原
            # J_recon = tf.reduce_sum(tf.square(embedding_yp_orisample - embedding))   # 1024维
            # J_recon = (1 - FLAGS._lambda) * tf.reduce_sum(tf.abs(embedding_yp - embedding_y_origin))

            # J_recon = tf.reduce_mean(1 - cosine(embedding_yp_orisample, embedding_y_origin)) + 0.5*noise # 改为cosin相似度
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
            J_gen = J_recon

    if FLAGS.Apply_HDML:
        # train_step = nn_Ops.training(loss=J_m, lr=lr)
        # """定义指数衰减学习率    #10为过完所有训练数据需要的迭代次数   staircase=False为连续曲线状衰减，为true时为阶梯状衰减"""
        # 这个参数设置为True时，则只有在迭代完1000次时（即decay_step的值），学习率才会乘上衰减系数0.95；
        # 设置为False时，则在每一次迭代完成时，学习率都会更新，相当于按照公式实时更新。
        # metric_learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step1, decay_step*10,
        #                                                   FLAGS.decay_rate, staircase=True)
        # #
        # gen_learning_rate = tf.train.exponential_decay(FLAGS.lr_gen, global_step2, decay_step*10, FLAGS.decay_rate, staircase=True)
        # metric_learning_rate = tf.train.polynomial_decay(
        #     learning_rate=FLAGS.init_learning_rate, global_step=global_step1, decay_steps=6000,
        #     end_learning_rate=0.00005, power=0.5, cycle=False)

        c_train_step = nn_Ops.training(loss=J_metric, lr=FLAGS.init_learning_rate, var_scope='Classifier', global_step=global_step1)
        g_train_step = nn_Ops.training(loss=J_gen, lr=FLAGS.lr_gen, var_scope='Generator', global_step=global_step2)
        # s_train_step = nn_Ops.training(loss=cross_entropy, lr=FLAGS.s_lr, var_scope='Softmax_classifier')
    else:
        # metric_learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step1, decay_step,
        #                                                   FLAGS.decay_rate, staircase=True)
        train_step = nn_Ops.training(loss=J_m, lr=FLAGS.init_learning_rate, global_step=global_step1)

    # initialise the session
    with tf.Session(config=config) as sess:
        # Initial all the variables with the sess

        sess.run(tf.global_variables_initializer())
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline", dump_root="D:/model/debug")
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        saver = tf.train.Saver()

        # learning rate
        # _lr = FLAGS.init_learning_rate

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
                    train, J_m_var, = sess.run([train_step, J_m],
                                                           feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                                                                      is_Training: True})
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

                    c_train, g_train,  J_metric_var, J_m_var, \
                    J_syn_var, J_recon_var, J_gen_var, noise_var, _label_raw,\
                        _I_pos, _positive_emb, _I_neg, _negative_emb, _syn_factor, \
                    wd_Loss_var,_global_step1,_global_step2 = sess.run(   #
                        [c_train_step, g_train_step,
                         J_metric, J_m, J_syn, J_recon, J_gen, noise,  label_raw,
                        I_pos, positive_emb, I_neg, negative_emb, syn_factor,
                         wdLoss,global_step1,global_step2],  #
                        feed_dict={x_raw: x_batch_data,
                                   label_raw: Label_raw,
                                   is_Training: True, Javg: J_m_loss.read(), Jgen: J_gen_loss.read()})

                    pbar.set_description("J_metric_var: %f, Jm: %f, J_syn_var: %f, "
                                         "J_recon_var: %f, noise_var: %f, J_gen_var: %f, syn_factor: %f, wdLoss: %f, global_step1: %d, "
                                         "global_step2: %d"%
                                         (J_metric_var, J_m_var, J_syn_var, J_recon_var, noise_var, J_gen_var,
                                          _syn_factor, wd_Loss_var, _global_step1,_global_step2))

                    wd_Loss.update(var=wd_Loss_var)
                    J_metric_loss.update(var=J_metric_var)
                    J_m_loss.update(var=J_m_var)
                    J_syn_loss.update(var=J_syn_var)
                    J_recon_loss.update(var=J_recon_var)

                    J_gen_loss.update(var=J_gen_var)

                step += 1
                # print('learning rate %f' % _lr)

                # evaluation
                if step % bp_epoch == 0:  # 6个epoch验证一次
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
                    # eval_summary.value.add(tag='learning_rate', simple_value=_lr)
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
                        print("Saved with max_f1_score:", max_f1_score, "-->", LOGDIR)
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                    summary_writer.flush()
                    # if step in [5632, 6848]:  # 原来的
                    # if step % 1000:
                    #     _lr = _lr * 0.9

                    if step >= 5000:
                        bp_epoch = FLAGS.batch_per_epoch
                    if step >= FLAGS.max_steps:
                        os._exit()

if __name__ == '__main__':
    tf.app.run()
