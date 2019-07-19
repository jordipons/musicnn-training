import tensorflow as tf
import numpy as np
import autopool as ap


def temporal_pooling(feature_map, is_training, num_classes_dataset, num_units_backend, type):

    # which temporal pooling?

    if 'rnn' in type: 
        print('Recurrent Neural Networks')
        reshaped = tf.squeeze(feature_map, [3])
        rnn = tf.keras.layers.CuDNNLSTM(512, return_sequences=True, name='lstm1')(reshaped)
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
        padded = tf.pad(feature_map, [[0, 0], [int(context/2), int(context/2)], [0, 0], [0, 0]], "CONSTANT")
        frames_attention = tf.layers.conv2d(inputs=padded,
                             filters=padded.shape[2],
                             kernel_size=[context, padded.shape[2]],
                             padding="valid",
                             activation=None,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        softmax_layer = tf.nn.softmax(frames_attention,axis=1)

        # apply attention
        frames_attention_t = tf.transpose(softmax_layer, [0, 1, 3, 2])
        weighted = tf.multiply(frames_attention_t, feature_map)

        # temporal pooling
        tmp_pool = tf.reduce_sum(weighted, axis=1)

    elif 'autopool' in type:
        print('Auto-pool')
        tmp_pool = ap.AutoPool1D(axis=1)(feature_map)
        
    else:
        print('Max/Avg pooling')
        max_pool = tf.reduce_max(feature_map, axis=1)
        avg_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
        tmp_pool = tf.concat([max_pool, avg_pool], 2)

    # dense layer with droupout
    flat_pool = tf.contrib.layers.flatten(tmp_pool)
    flat_pool = tf.layers.batch_normalization(flat_pool, training=is_training)
    flat_pool_dropout = tf.layers.dropout(flat_pool, rate=0.5, training=is_training)
    dense = tf.layers.dense(inputs=flat_pool_dropout,
                            units=num_units_backend,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_dense = tf.layers.batch_normalization(dense, training=is_training)

    # output layer
    dense_dropout = tf.layers.dropout(bn_dense, rate=0.5, training=is_training)
    return tf.layers.dense(inputs=dense_dropout,
                           activation=None,
                           units=num_classes_dataset,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer())


def positional_encoding(feature_map_size):
    _, T, num_units, _ = feature_map_size
    freq = 10000 # 10000 or 5
    position_enc = np.array([[pos / np.power(freq, 2.*i/num_units) for i in range(num_units)] for pos in range(T)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    outputs = tf.convert_to_tensor(position_enc,dtype=tf.float32)
    #outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
    return tf.reshape(outputs,[-1,T,num_units,1])

