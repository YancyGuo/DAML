from datasets import data_provider
# import datasets.data_provider as data_provider
from lib import GoogleNet_Model, Loss_ops, nn_Ops, Embedding_Visualization, HDML_n_pnt2, evaluation
import copy
from tqdm import tqdm
from tensorflow.contrib import layers
from flags.FLAGS_HDML_triplet import *

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
dataset_splite_num = FLAGS.path.split('/')[-1].split("_")[-1].split(".")[0]
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'_'+'Triplet_'+dataset_splite_num+'/'


if FLAGS. SaveVal:
    nn_Ops.create_path(_time)
summary_writer = tf.summary.FileWriter(LOGDIR)


def main(_):
    if not FLAGS.LossType == 'easy_pos_hard_negLoss':
        print("LossType easy_pos_hard_negLoss loss is required")
        return 0

    # placeholders
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)

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
            #embedding_l = embedding_z
            J_m = Loss_ops.Loss(embedding_z, label, FLAGS.LossType) + wdLoss

    # if HNG is applied
    if FLAGS.Apply_HDML:

        with tf.name_scope('Javg'):
            Javg = tf.placeholder(tf.float32)
        with tf.name_scope('Jrecon_var'):
            Jrecon_var = tf.placeholder(tf.float32)

        # 在嵌入空间合成的 hardness-aware tuple [anc, pos, neg]  (主要是合成了neg)
        # nan是因为生成的样本使reuse的y-->z的全连接层出现问题了
        syn_embedding_z, syn_sample_label = HDML_n_pnt2.Pulling(FLAGS.LossType, embedding_z, label)  # 原来javg
        # embedding_z_quta = tf.Print(embedding_z_quta, [embedding_z_quta], "embedding_z_quta", summarize=100)

        # 原来的和合成的tuple合并
        embedding_z_concate = tf.concat([embedding_z, syn_embedding_z], axis=0)

        # Generator
        # 将原样本与合成样本都映射回feature space
        # z--->y’,z^--->y~ ,maps the augmented embeding of a tuple back to the feature space.
        # maps not only the synthetic negative sample but other unaltered samples in one tuple.
        with tf.variable_scope('Generator'):
            # generator fc3
            embedding_y_concate = nn_Ops.fc_block(
                embedding_z_concate, in_d=FLAGS.embedding_size, out_d=512,
                name='generator1', is_bn=True, is_relu=True, is_Training=is_Training
            )

            # generator fc4???
            embedding_y_concate = nn_Ops.fc_block(
                embedding_y_concate, in_d=512, out_d=1024,
                name='generator2', is_bn=False, is_relu=False, is_Training=is_Training
            )

            #  embedding_yp是原样本映射回去的， 而embedding_yq是合成样本映射的
            embedding_yp, embedding_yq1, embedding_yq2 = tf.split(embedding_y_concate, 3, axis=0)
            embedding_yq = tf.concat([embedding_yq1, embedding_yq2], axis=0)

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

            syn_factor = tf.exp(-FLAGS.beta / Jrecon_var)
            J_syn = (1- syn_factor)*Loss_ops.Loss(embedding_z_quta, syn_sample_label, 0, _lossType=FLAGS.LossType)
            J_m2 = syn_factor * J_m
            J_metric = J_m2 + J_syn  # 总的度量学习损失


            # J_syn = Loss_ops.Loss(embedding_z_quta, syn_sample_label, 0, _lossType=FLAGS.LossType)
            # J_metric = J_m + J_syn  # 总的度量学习损失

            # 重建损失Jrecon = ||y − y′||2 2 to restrict the encoder & decoder to
            # map each point close to itself.
            # 因为y----encoder--->z----decoder(生成器)--->y'，所以要使映射回feature space的y'接近它本身
            class0_emb1, class0_emb2, class1_emb= tf.split(embedding_y_origin, 3, axis=0)
            class0_emb = tf.concat([class0_emb1, class0_emb2], axis=0)
            emb0_permute = tf.gather(class0_emb, tf.random_shuffle(tf.range(tf.shape(class0_emb)[0])))
            emb1_permute = tf.gather(class1_emb, tf.random_shuffle(tf.range(tf.shape(class1_emb)[0])))
            embedding_y_origin_permute = tf.concat([emb0_permute, emb1_permute], axis=0)

            noise = tf.reduce_mean(tf.square(embedding_yp - embedding_y_origin_permute))
            J_recon = (1 - FLAGS._lambda) * tf.reduce_sum(tf.square(embedding_yp - embedding_y_origin)) + noise

            # 使用原始的特征y训练全连接层，然后用训练好的参数用来保证生成样本标签一致。
            # we simultaneously learn a fully connected layer with the softmax loss on y,
            # where the gradients only update the parameters in this layer.
            # We employ the learned softmax layer to compute the softmax loss
            # jsoft(y~, l) between the synthetic hardness-aware negative y~
            # and the original label l.

    if FLAGS.Apply_HDML:
        # train_step = nn_Ops.training(loss=J_m, lr=lr)
        c_train_step = nn_Ops.training(loss=J_metric, lr=lr, var_scope='Classifier')
        g_train_step = nn_Ops.training(loss=J_recon, lr=FLAGS.lr_gen, var_scope='Generator')
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
        epoch_iterator = streams_no_aug.get_epoch_iterator()

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
                    train, J_m_var, wd_Loss_var = sess.run([train_step, J_m, wdLoss],
                                                           feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                                                                      is_Training: True, lr: _lr})
                    # train, J_m_var, = sess.run([train_step, J_m],
                    #                                        feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                    #                                                   is_Training: True, lr: _lr})
                    J_m_loss.update(var=J_m_var)
                    wd_Loss.update(var=wd_Loss_var)
                    pbar.set_description("Jm: %f, wd_Loss_var: %f" % (J_m_var, wd_Loss_var))


                else:
                    # summaries = tf.summary.merge_all()
                    # c_train, g_train, s_train, wd_Loss_var, J_metric_var, J_m_var, \
                    #         J_syn_var, J_recon_var, J_soft_var, J_gen_var, cross_en_var, _label_raw, summ, neg2_, neg_, neg_mask_ = sess.run(
                    #         [c_train_step, g_train_step, s_train_step, wdLoss,
                    #          J_metric, J_m, J_syn, J_recon, J_soft, J_gen, cross_entropy, label_raw, summaries, neg2, neg, neg_mask],
                    #         feed_dict={x_raw: x_batch_data,
                    #                    label_raw: Label_raw,
                    #                    is_Training: True, lr: _lr, Javg: J_m_loss.read(), Jgen: J_gen_loss.read()})

                    c_train, g_train, wd_Loss_var, J_metric_var, J_m_var, \
                    J_syn_var, J_recon_var, _label_raw, _syn_factor = sess.run(
                        [c_train_step, g_train_step, wdLoss,
                         J_metric, J_m, J_syn, J_recon, label_raw, syn_factor],
                        feed_dict={x_raw: x_batch_data,
                                   label_raw: Label_raw,
                                   is_Training: True, lr: _lr, Javg: J_m_loss.read(), Jrecon_var: J_recon_loss.read()})
                    pbar.set_description("J_metric_var: %f, Jm: %f, J_syn_var: %f, "
                                         "J_recon_var: %f, syn_factor: %f, wdLoss: %f" %
                                         (J_metric_var, J_m_var, J_syn_var, J_recon_var, _syn_factor,
                                          wd_Loss_var))

                    wd_Loss.update(var=wd_Loss_var)
                    J_metric_loss.update(var=J_metric_var)
                    J_m_loss.update(var=J_m_var)
                    J_syn_loss.update(var=J_syn_var)
                    J_recon_loss.update(var=J_recon_var)

                step += 1
                # print('learning rate %f' % _lr)

                # evaluation
                if step % bp_epoch == 0:
                    print('only eval eval')
                    # nmi_tr, f1_tr, recalls_tr = evaluation.Evaluation(
                    #     stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)
                    # nmi_te, f1_te, recalls_te = evaluation.Evaluation(
                    #     stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 4, neighbours, logits, label)
                    # nmi_te, f1_te, recalls_te = evaluation.Evaluation(
                    #         stream_test, stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 2, neighbours)
                    nmi_te, f1_te, recalls_te, recall, precision, f1_score, max_f1score = evaluation.Evaluation(
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
                    summary_writer.add_summary(eval_summary, step)
                    print('Summary written')
                    if f1_score > max_f1_score:
                        max_f1_score = f1_score

                        print("Saved with max_f1_score:", max_f1_score, "--->", LOGDIR)
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                    summary_writer.flush()
                    if step in [5632, 6848]:  # 原来的
                    # if step in [1500, 6848]:
                        _lr = _lr * 0.9

                    if step >= 5000:
                        bp_epoch = FLAGS.batch_per_epoch
                    if step >= FLAGS.max_steps:
                        os._exit()

if __name__ == '__main__':
    tf.app.run()
