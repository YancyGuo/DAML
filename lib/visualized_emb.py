import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def embedding_assign(batch_size, embedding, embedding_size, name):
    """
    创建embedding_var变量，并赋值
    :param batch_size:
    :param embedding:
    :param embedding_size:
    :param name:
    :return: 返回赋值操作assignment，及变量embedding_var
    """
    # 创建变量 shape为[batch_size, embedding_size]
    embedding_var = tf.Variable(tf.zeros([batch_size, embedding_size]),
                                name=name, trainable=False)
    # 创建操作, 将embedding赋值给embedding_var
    assignment = embedding_var.assign(embedding)
    return assignment, embedding_var


def embedding_Visual(embedding_var, tensor_name, LOGDIR, META_FILE, SPRITE_FILE, TRAINING_STEPS):
    with tf.Session() as sess:
        # 使用新的变量来保存模型输出的embedding
        y = tf.Variable(embedding_var, name=tensor_name)
        summary_writer = tf.summary.FileWriter(LOGDIR)

        # 通过projector.ProjectorConfig类来帮助生成日志文件
        config = projector.ProjectorConfig()
        # 增加一个需要可视化的embedding结果
        embedding_config = config.embeddings.add()
        # 指定这个embedding结果对应的tensorflow变量名称
        embedding_config.tensor_name = y.name
        # 指定对应的原始数据信息
        embedding_config.metadata_path = META_FILE
        # 指定sprite图像
        embedding_config.sprite.image_path = SPRITE_FILE
        # 从sprite图像截取正确的原始图像
        embedding_config.sprite.single_image_dim.extend([32, 32])

        # 写入日志
        projector.visualize_embeddings(summary_writer, config)

        # 生成会话， 初始化新声明的变量并将需要的日志信息写入文件
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), TRAINING_STEPS)

        summary_writer.close()
