import tensorflow as tf
import models_transformer


def backend(route_out, is_training, config, num_filt, output_units, type):
    concat_front_end = midend(route_out, is_training, config, num_filt, type)
    return temporal_pool(concat_front_end, is_training, config, output_units, type)


def midend(route_out, is_training, config, num_filt, type):
    route_out = tf.expand_dims(route_out, 3)
    print('Input to backend: ' + str(route_out.get_shape))

    # conv layer 1 - adapting dimensions
    route_out_pad = tf.pad(route_out, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv1 = tf.layers.conv2d(inputs=route_out_pad,
                             filters=num_filt,
                             kernel_size=[7, route_out.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    if 'mp12' in type:
        print('Input to mp12: ' + str(bn_conv1_t.get_shape))
        bn_conv1_t = tf.layers.max_pooling2d(inputs=bn_conv1_t, pool_size=[2, 1], strides=[2, 1], name='pool12')
        print('Out to mp12: ' + str(bn_conv1_t.get_shape))

    if 'mp441' in type:
        print('Input to mp441: ' + str(bn_conv1_t.get_shape))
        bn_conv1_t = tf.layers.max_pooling2d(inputs=bn_conv1_t, pool_size=[4, 1], strides=[4, 1], name='pool12')
        print('Out to mp441: ' + str(bn_conv1_t.get_shape))

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.layers.conv2d(inputs=bn_conv1_pad,
                             filters=num_filt,
                             kernel_size=[7, bn_conv1_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.transpose(bn_conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2, bn_conv1_t)

    if 'mp23' in type:
        print('Input to mp23: ' + str(res_conv2.get_shape))
        res_conv2 = tf.layers.max_pooling2d(inputs=res_conv2, pool_size=[2, 1], strides=[2, 1], name='pool23')
        print('Out to mp23: ' + str(res_conv2.get_shape))

    if 'mp442' in type:
        print('Input to mp442: ' + str(res_conv2.get_shape))
        res_conv2 = tf.layers.max_pooling2d(inputs=res_conv2, pool_size=[4, 1], strides=[4, 1], name='pool23')
        print('Out to mp442: ' + str(res_conv2.get_shape))

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv3 = tf.layers.conv2d(inputs=bn_conv2_pad,
                             filters=num_filt,
                             kernel_size=[7, bn_conv2_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.transpose(bn_conv3, [0, 1, 3, 2])
    res_conv3 = tf.add(conv3, res_conv2)

    # which layers?
    if 'dense' in type:
        concat_front_end = tf.concat([route_out, bn_conv1_t, res_conv2, res_conv3], 2)
        print('Dense')
    else:
        concat_front_end = res_conv3
        print('NO Dense')

    print('concat_front_end',concat_front_end.get_shape())
    return concat_front_end


def temporal_pool(feature_map, is_training, config, output_units, type):

    # which temporal pooling?
    if 'rnn' in type: 
        if '32' in type:
            num_rnn_units=32
        elif '128' in type:
            num_rnn_units=128
        elif '512' in type:
            num_rnn_units=512
        elif '1024' in type:
            num_rnn_units=1024
        print('Recurrent Neural Networks')
        reshaped = tf.squeeze(feature_map, [3])
        rnn = tf.keras.layers.CuDNNLSTM(num_rnn_units, return_sequences=True, name='lstm1')(reshaped)
        tmp_pool = tf.keras.layers.CuDNNLSTM(num_rnn_units, return_sequences=False, name='lstm2')(rnn)
        print('rnn',rnn.get_shape())
        print('reshaped',reshaped.get_shape())

    elif 'attention' in type: 
        print('Attention')
        if 'position' in type:
            if 'sin' in type:
                pos_embedding = models_transformer.position_embedding_sin(feature_map.get_shape().as_list())
            else:
                pos_embedding = models_transformer.position_embedding(feature_map.get_shape().as_list(),is_training)
            print('Position embedding', pos_embedding.get_shape())
            feature_map = tf.add(feature_map, pos_embedding)
            print('After position + front_end',feature_map.get_shape())

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

        print('feature_map',feature_map.get_shape())
        print('padded',padded.get_shape())
        print('frames_attention',frames_attention.get_shape())
        print('softmax_layer',softmax_layer.get_shape())
        print('frames_attention_t',frames_attention_t.get_shape())
        print('weighted',weighted.get_shape())
        print('tmp_pool',tmp_pool.get_shape())

    elif 'autopool' in type:
        print('Auto-pool')
        import autopool as ap
        if 'position' in type:
            if 'sin' in type:
                pos_embedding = models_transformer.position_embedding_sin(feature_map.get_shape().as_list())
            else:
                pos_embedding = models_transformer.position_embedding(feature_map.get_shape().as_list(),is_training)
            print('Position embedding', pos_embedding.get_shape())
            feature_map = tf.add(feature_map, pos_embedding)
            print('After position + front_end',feature_map.get_shape())
        tmp_pool = ap.AutoPool1D(axis=1)(feature_map)
        
    else:
        print('Max/Avg pooling')
        max_pool = tf.reduce_max(feature_map, axis=1)
        avg_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
        tmp_pool = tf.concat([max_pool, avg_pool], 2)

    # output - 2 dense layer with droupout
    flat_pool = tf.contrib.layers.flatten(tmp_pool)
    print('flat_pool', flat_pool.get_shape())
    flat_pool = tf.layers.batch_normalization(flat_pool, training=is_training) # <-- batch norm here? It improved with batch normalization
    flat_pool_dropout = tf.layers.dropout(flat_pool, rate=0.5, training=is_training)
    dense = tf.layers.dense(inputs=flat_pool_dropout,
                            units=output_units,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_dense = tf.layers.batch_normalization(dense, training=is_training)
    dense_dropout = tf.layers.dropout(bn_dense, rate=0.5, training=is_training)
    return tf.layers.dense(inputs=dense_dropout,
                           activation=None,
                           units=config['num_classes_dataset'],
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

