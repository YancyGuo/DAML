from lib import GoogleNet_Model, nn_Ops, HDML
from flags.FLAGS_HDML_triplet import *


def inference(x_raw, is_Training):
    # CNN 分类器
    with tf.variable_scope('Classifier'):
        google_net_model = GoogleNet_Model.GoogleNet_Model()
        embedding = google_net_model.forward(x_raw)  # 输入数据经过GoogleNet得到嵌入

        # 如果使用 hard-aware Negative Generation，则记录原始特征y（特征空间）
        if FLAGS.Apply_HDML:
            # Tensor("Classifier/Reshape_1:0", shape=(?, 1024), dtype=float32)
            embedding_y_origin = embedding

        # Batch Normalization layer 1 批量归一化
        # Tensor("Classifier/BN1/batch_normalization/batchnorm/add_1:0", shape=(?, 1024), dtype=float32)
        embedding = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1')
        # FC layer 1  全连接 得到嵌入空间中的Z Tensor("Classifier/fc1/add:0", shape=(?, 128), dtype=float32)
        embedding_z = nn_Ops.fc_block(
            embedding, in_d=1024, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training)

        # Embedding Visualization  嵌入可视化
        # 赋值op 和变量embedding_var
        # assignment, embedding_var = Embedding_Visualization.embedding_assign(
        #     batch_size=256, embedding=embedding_z,
        #     embedding_size=FLAGS.embedding_size, name='Embedding_of_fc1')

    # if HNG is applied 如果使用Hard Negative Mining
    if FLAGS.Apply_HDML:
        # 占位符
        # 插值生成Z^ Tensor("concat:0", shape=(72, 128), dtype=float32)
        embedding_z_quta = HDML.Pulling(FLAGS.LossType, embedding_z, Javg)
        # Tensor("concat_1:0", shape=(?, 128), dtype=float32)
        embedding_z_concate = tf.concat([embedding_z, embedding_z_quta], axis=0)

        # Generator 把z映射回y  128-->1024
        with tf.variable_scope('Generator'):
            # generator fc3
            embedding_y_concate = nn_Ops.fc_block(
                embedding_z_concate, in_d=FLAGS.embedding_size, out_d=512,
                name='generator1', is_bn=True, is_relu=True, is_Training=is_Training
            )

            # generator fc4  Tensor("Generator/generator2/add:0", shape=(?, 1024), dtype=float32)
            embedding_y_concate = nn_Ops.fc_block(
                embedding_y_concate, in_d=512, out_d=1024,
                name='generator2', is_bn=False, is_relu=True, is_Training=is_Training
            )

            # Tensor("Generator/Slice:0", shape=(16, 1024), dtype=float32)
            # Tensor("Generator/Slice_1:0", shape=(72, 1024), dtype=float32)
            embedding_yp, embedding_yq = HDML.npairSplit(embedding_y_concate)

        # 用嵌入z z^训练和上面一样的分类层
        with tf.variable_scope('Classifier'):
            embedding_z_quta = nn_Ops.bn_block(
                embedding_yq, normal=FLAGS.normalize, is_Training=is_Training, name='BN1', reuse=True)

            embedding_z_quta = nn_Ops.fc_block(  # (72,128)
                embedding_z_quta, in_d=1024, out_d=FLAGS.embedding_size,
                name='fc1', is_bn=False, is_relu=False, reuse=True, is_Training=is_Training
            )

            # 划分anchor neg
            # (8,128)
            embedding_zq_anc = tf.slice(
                input_=embedding_z_quta, begin=[0, 0], size=[int(FLAGS.batch_size / 2), int(FLAGS.embedding_size)])
            # (64,128)  生成的
            embedding_zq_negtile = tf.slice(
                input_=embedding_z_quta, begin=[int(FLAGS.batch_size / 2), 0],
                size=[int(np.square(FLAGS.batch_size / 2)), int(FLAGS.embedding_size)]
            )
        return embedding_z, embedding_y_origin, embedding_zq_anc, embedding_zq_negtile, embedding_yp, embedding_yq
    else:
        return embedding_z