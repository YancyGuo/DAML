from datasets import data_provider
from lib import evaluation
from flags.FLAGS_HDML_triplet import *

# Using GPU
from prostate_inference import inference

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def prostate_evaluate(stream_train_eval, stream_test):
    # placeholders Tensor("Placeholder:0", shape=(?, 227, 227, 3), dtype=float32)
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    # Tensor("Placeholder_1:0", shape=(?, 1), dtype=int32)
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)

    # 直接通过调用封装好的函数来计算前向传播结果。
    if FLAGS.Apply_HDML:
        embedding_z, embedding_y_origin, embedding_zq_anc, \
        embedding_zq_negtile, embedding_yp, embedding_yq = inference(x_raw, is_Training)
    else:
        embedding_z = inference(x_raw, is_Training)

    # initialise the session
    with tf.Session(config=config) as sess:
        # Initial all the variables with the sess
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,
                      "/home/ZSL/workspace/HDML/tensorboard_log/prostate/easy_pos_hard_negLoss/04-05-22-26用了hdml/model.ckpt")
        # x_batch_data, c_batch_data = copy.copy(stream_train.get_epoch_iterator())
        # _embedding_z = sess.run(embedding_z, feed_dict={x_raw: x_batch_data, is_Training: True})

        # evaluation
        print('only eval eval')
        # nmi_tr, f1_tr, recalls_tr = evaluation.Evaluation(
        # stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)
        # nmi_te, f1_te, recalls_te = evaluation.Evaluation(
        #     stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 98, neighbours)
        nmi_te, f1_te, recalls_te, recall, precision, f1_score = evaluation.Evaluation(
            stream_test, stream_train_eval, image_mean, sess, x_raw, label_raw, is_Training, embedding_z, 2,
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


def main(argv=None):
    # Create the stream of datas from dataset
    streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, method, crop_size=FLAGS.default_image_size)
    stream_train, stream_train_eval, stream_test = streams
    prostate_evaluate(stream_train_eval, stream_test)


if __name__ == '__main__':
    tf.app.run()