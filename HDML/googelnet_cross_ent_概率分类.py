from datasets import data_provider
# import datasets.data_provider as data_provider
from lib import GoogleNet_Model, nn_Ops, HDML, evaluation
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
dataset_splite_num = FLAGS.path.split('/')[-1].split("_")[-1].split(".")[0]
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'_'+'CE_'+dataset_splite_num+'/'

if FLAGS. SaveVal:
    nn_Ops.create_path(_time)
summary_writer = tf.summary.FileWriter(LOGDIR)


def main(_):
    if not FLAGS.LossType == 'cross_entropy':
        print("LossType cross_entropy loss is required")
        return 0
    if FLAGS.Apply_HDML:
        print("不要用HDML")
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
        # Batch Normalization layer 1
        embedding = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1')
        # FC layer 1
        embedding_z = nn_Ops.fc_block(                        # 256
            embedding, in_d=1024, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)


        # Get the Label
        label = tf.reduce_mean(label_raw, axis=1, keep_dims=False)
        # For some kinds of Losses, the embedding should be l2 normed
        #embedding_l = embedding_z


    # if HNG is applied
    # if FLAGS.Apply_HDML:
    with tf.name_scope('Loss'):
        cross_entropy, W_fc, b_fc, logits = HDML.cross_entropy(embedding=embedding_z, label=label,size=256)
        predict_labels = tf.arg_max(logits, 1)

    # if FLAGS.Apply_HDML:
    #     s_train_step = nn_Ops.training(loss=cross_entropy, lr=FLAGS.s_lr, var_scope='Softmax_classifier')
    # else:
    train_step = nn_Ops.training(loss=cross_entropy, lr=lr)

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

        # collectors
        # J_m_loss = nn_Ops.data_collector(tag='Jm', init=1e+6)  # 空
        # J_syn_loss = nn_Ops.data_collector(tag='J_syn', init=1e+6)
        # J_metric_loss = nn_Ops.data_collector(tag='J_metric', init=1e+6)  # 空
        # J_soft_loss = nn_Ops.data_collector(tag='J_soft', init=1e+6)
        # J_recon_loss = nn_Ops.data_collector(tag='J_recon', init=1e+6)
        # J_gen_loss = nn_Ops.data_collector(tag='J_gen', init=1e+6)
        cross_entropy_loss = nn_Ops.data_collector(tag='cross_entropy', init=1e+6)
        # wd_Loss = nn_Ops.data_collector(tag='weight_decay', init=1e+6)
        max_nmi = 0
        max_f1_score = 0
        step = 0

        r_step = []
        r_mask = []
        r_neg = []
        r_neg2 = []

        bp_epoch = FLAGS.init_batch_per_epoch
        epoch_iterator = streams_no_aug.get_epoch_iterator()
        with tqdm(total=FLAGS.max_steps) as pbar:
            for batch in copy.copy(epoch_iterator):
                # get images and labels from batch
                x_batch_data, Label_raw = nn_Ops.batch_data(batch)
                pbar.update(1)
                if not FLAGS.Apply_HDML:
                    train, cross_en_var = sess.run([train_step, cross_entropy],
                                                           feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                                                                      is_Training: True, lr: _lr})
                    # train, J_m_var, = sess.run([train_step, J_m],
                    #                                        feed_dict={x_raw: x_batch_data, label_raw: Label_raw,
                    #                                                   is_Training: True, lr: _lr})
                    cross_entropy_loss.update(cross_en_var)
                    pbar.set_description("cross_entropy_loss: %f" % (cross_en_var))

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
                    #
                    # knn_nmi_te, knn_f1_te, knn_recalls_te, knn_recall, knn_precision, knn_f1_score, knn_max_f1score = evaluation.Evaluation(
                    #     stream_test, stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training,
                    #     embedding_z, 2,
                    #     neighbours)

                    nmi_te, f1_te, recalls_te, recall, precision, f1_score, k_max_f1score = evaluation.Evaluation_with_crossent(
                        stream_test,
                        image_mean, sess, x_raw, label_raw,
                        is_Training, embedding_z, predict_labels, origin_label=label)

                    print("nmi: %f, f1: %f" % (nmi_te,f1_te) )

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
                    # for i in range(0, np.shape(neighbours)[0]):
                    #     eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te[i])
                    eval_summary.value.add(tag='Recall', simple_value=recalls_te)
                    cross_entropy_loss.write_to_tfboard(eval_summary)
                    # wd_Loss.write_to_tfboard(eval_summary)
                    eval_summary.value.add(tag='learning_rate', simple_value=_lr)

                        # prior_jm_loss.write_to_tfboard(eval_summary)  # 33
                    summary_writer.add_summary(eval_summary, step)
                    print('Summary written')
                    if k_max_f1score > max_f1_score:
                        max_f1_score = k_max_f1score
                        print("Saved with max_f1_score:", max_f1_score, "-->", LOGDIR)
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                    summary_writer.flush()
                    # if step in [5632, 6848]:  # 原来的
                    # # if step in [1500, 6848]:
                    #     _lr = _lr * 0.5

                    if step >= 5000:
                        bp_epoch = FLAGS.batch_per_epoch
                    if step >= FLAGS.max_steps:
                        os._exit()


if __name__ == '__main__':
    tf.app.run()
