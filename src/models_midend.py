import tensorflow as tf


def dense_cnns(front_end_output, is_training, num_filt):

    # conv layer 1 - adapting dimensions
    front_end_pad = tf.pad(front_end_output, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv1 = tf.compat.v1.layers.conv1d(inputs=front_end_pad,
                             filters=num_filt,
                             kernel_size=7,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training)

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv2 = tf.compat.v1.layers.conv1d(inputs=bn_conv1_pad,
                             filters=num_filt,
                             kernel_size=7,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training)
    res_conv2 = tf.add(conv2, bn_conv1)

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv3 = tf.compat.v1.layers.conv1d(inputs=bn_conv2_pad,
                             filters=num_filt,
                             kernel_size=7,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training)
    res_conv3 = tf.add(conv3, res_conv2)

    return [front_end_output, bn_conv1, res_conv2, res_conv3]

