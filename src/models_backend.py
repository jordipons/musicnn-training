import tensorflow as tf
import numpy as np


def temporal_pooling(feature_map, is_training, num_classes_dataset, num_units_backend, type):

    # which temporal pooling?

    if 'rnn' in type: 
        print('Recurrent Neural Networks')
        rnn = tf.keras.layers.CuDNNLSTM(512, return_sequences=True, name='lstm1')(feature_map)
        tmp_pool = tf.keras.layers.CuDNNLSTM(512, return_sequences=False, name='lstm2')(rnn)

    elif 'attention' in type: 
        print('Attention')

        # add positional encoding

        if 'positional' in type:
            print('Using positional encoding')
            pos_embedding = positional_encoding(feature_map.get_shape().as_list())
            feature_map = tf.add(feature_map, pos_embedding)

        # compute attention
        context=3
        padded = tf.pad(feature_map, [[0, 0], [int(context/2), int(context/2)], [0, 0]], "CONSTANT")
        frames_attention = tf.compat.v1.layers.conv1d(inputs=padded,
                             filters=padded.shape[2],
                             kernel_size=context,
                             padding="valid",
                             activation=None,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        softmax_layer = tf.nn.softmax(frames_attention,axis=1)

        # apply attention
        weighted = tf.multiply(softmax_layer, feature_map)

        # temporal pooling
        tmp_pool = tf.reduce_sum(weighted, axis=1)

    elif 'autopool' in type:
        print('Auto-pool')
        alpha = tf.Variable(tf.constant(0, dtype=tf.float32), name='alpha', trainable=True)
        # alpha initialized to 0, which is the safest option. 1 could also be an interesting initialization!
        scaled = tf.scalar_mul(alpha,feature_map)
        max_val = tf.reduce_max(scaled, axis=1, keepdims=True)
        softmax = tf.exp(scaled - max_val)
        weights = softmax / tf.reduce_sum(softmax, axis=1, keepdims=True)
        tmp_pool = tf.reduce_sum(feature_map * weights, axis=1, keepdims=False)

    else:
        print('Max/Avg pooling')
        max_pool = tf.reduce_max(feature_map, axis=1)
        avg_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
        tmp_pool = tf.concat([max_pool, avg_pool], 1)

    print('Temporal pooling: ' + str(tmp_pool.shape))
    # dense layer with droupout
    flat_pool = tf.contrib.layers.flatten(tmp_pool)
    flat_pool = tf.compat.v1.layers.batch_normalization(flat_pool, training=is_training)
    flat_pool_dropout = tf.layers.dropout(flat_pool, rate=0.5, training=is_training)
    dense = tf.compat.v1.layers.dense(inputs=flat_pool_dropout,
                            units=num_units_backend,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_dense = tf.compat.v1.layers.batch_normalization(dense, training=is_training)

    # output layer
    dense_dropout = tf.compat.v1.layers.dropout(bn_dense, rate=0.5, training=is_training)
    return tf.compat.v1.layers.dense(inputs=dense_dropout,
                           activation=None,
                           units=num_classes_dataset,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer())


def positional_encoding(feature_map_size):
    _, T, num_units = feature_map_size
    freq = 10000 # 10000 or 5
    position_enc = np.array([[pos / np.power(freq, 2.*i/num_units) for i in range(num_units)] for pos in range(T)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    outputs = tf.convert_to_tensor(position_enc,dtype=tf.float32)
    return tf.reshape(outputs,[-1,T,num_units])

