import tensorflow as tf

FREQ_AXIS=96
NUM_CLASSES_DATASET=50

def frontend(x, is_training, num_filt):

    print('input: ' + str(x.get_shape))
    expanded_layer = tf.expand_dims(x, 3)
    input_layer = tf.layers.batch_normalization(expanded_layer, training=is_training)
    print('input_layer: ', input_layer.get_shape())

    # TIMBRAL FEATURES
    # smart padding!
    input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    input_pad_3 = tf.pad(input_layer, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")

    f79 = timbral_block(inputs=input_pad_7,
                   filters=int(num_filt),
                   kernel_size=[7, int(0.9 * FREQ_AXIS)],
                   is_training=is_training)

    f74 = timbral_block(inputs=input_pad_7,
                   filters=int(num_filt),
                   kernel_size=[7, int(0.4 * FREQ_AXIS)],
                   is_training=is_training)

    f39 = timbral_block(inputs=input_pad_3,
                   filters=int(num_filt*2),
                   kernel_size=[3, int(0.9 * FREQ_AXIS)],
                   is_training=is_training)

    f34 = timbral_block(inputs=input_pad_3,
                   filters=int(num_filt*2),
                   kernel_size=[3, int(0.4 * FREQ_AXIS)],
                   is_training=is_training)

    f19 = timbral_block(inputs=input_layer,
                   filters=int(num_filt*4),
                   kernel_size=[1, int(0.9 * FREQ_AXIS)],
                   is_training=is_training)

    f14 = timbral_block(inputs=input_layer,
                   filters=int(num_filt*4),
                   kernel_size=[1, int(0.4 * FREQ_AXIS)],
                   is_training=is_training)

    # TEMPORAL FEATURES
    # average pooling!
    avg_input = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, FREQ_AXIS],
                                        strides=[1, FREQ_AXIS])
    avg_input_rs = tf.squeeze(avg_input, [3])

    t165 = temporal_block(inputs=avg_input_rs,
                          filters=num_filt,
                          kernel_size=165,
                          is_training=is_training)

    t128 = temporal_block(inputs=avg_input_rs,
                          filters=num_filt*2,
                          kernel_size=128,
                          is_training=is_training)

    t64 = temporal_block(inputs=avg_input_rs,
                          filters=num_filt*4,
                          kernel_size=64,
                          is_training=is_training)

    t32 = temporal_block(inputs=avg_input_rs,
                          filters=num_filt*8,
                          kernel_size=32,
                          is_training=is_training)

    # concatenate all feature maps
    concatenation = tf.concat([f79, f74, f39, f34, f19, f14, t165, t128, t64, t32], 2)
    print('Output concatenation: ', concatenation.get_shape())
    return concatenation



def timbral_block(inputs, filters, kernel_size, is_training, padding="valid", activation=tf.nn.relu,
                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

    conv = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            activation=activation,
                            kernel_initializer=kernel_initializer)
    bn_conv = tf.layers.batch_normalization(conv, training=is_training)
    pool = tf.layers.max_pooling2d(inputs=bn_conv,
                                   pool_size=[1, bn_conv.shape[2]],
                                   strides=[1, bn_conv.shape[2]])
    out = tf.squeeze(pool, [2])
    print('Output timbral block: ', out.get_shape())
    return out


def temporal_block(inputs, filters, kernel_size, is_training, padding="same", activation=tf.nn.relu,
                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

    conv = tf.layers.conv1d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding=padding,
                              activation=activation,
                              kernel_initializer=kernel_initializer)
    out = tf.layers.batch_normalization(conv, training=is_training)
    print('Output tempo-variant block: ', out.get_shape())
    return out


def backend(route_out, is_training, num_filt, output_units):

    print('Input to backend: ' + str(route_out.get_shape))

    # conv layer 1 - adapting dimensions
    route_out_pad = tf.pad(route_out, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    print('Pre CNN1: ' + str(route_out_pad.get_shape))
    conv1 = tf.layers.conv1d(inputs=route_out_pad,
                             filters=num_filt,
                             kernel_size=7,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    print('Out CNN1: ' + str(bn_conv1.get_shape))

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv2 = tf.layers.conv1d(inputs=bn_conv1_pad,
                             filters=num_filt,
                             kernel_size=7,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    res_conv2 = tf.add(conv2, bn_conv2)
    print('Out residual CNN2: ' + str(res_conv2.get_shape))

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv3 = tf.layers.conv1d(inputs=bn_conv2_pad,
                             filters=num_filt,
                             kernel_size=7,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    feature_map = tf.add(res_conv2, bn_conv3)
    print('Out residual CNN3: ' + str(feature_map.get_shape))

    print('Max/Avg pooling')
    max_pool = tf.reduce_max(feature_map, axis=1)
    avg_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
    print('Out max: ' + str(max_pool.get_shape))
    print('Out avg: ' + str(avg_pool.get_shape))
    tmp_pool = tf.concat([max_pool, avg_pool], 1)
    print('Out tmp_pool: ' + str(tmp_pool.get_shape))

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
                           units=NUM_CLASSES_DATASET,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    return concat_front_end

#input_layer:  (?, 187, 96, 1)
#Output timbral block:  (?, 187, 8)
#Output timbral block:  (?, 187, 8)
#Output timbral block:  (?, 187, 16)
#Output timbral block:  (?, 187, 16)
#Output timbral block:  (?, 187, 32)
#Output timbral block:  (?, 187, 32)
#Output tempo-variant block:  (?, 187, 8)
#Output tempo-variant block:  (?, 187, 16)
#Output tempo-variant block:  (?, 187, 32)
#Output tempo-variant block:  (?, 187, 64)
#Output concatenation:  (?, 187, 232)
#Input to backend: <bound method Tensor.get_shape of <tf.Tensor 'model/concat:0' shape=(?, 187, 232) dtype=float32>>
#Pre CNN1: <bound method Tensor.get_shape of <tf.Tensor 'model/Pad_2:0' shape=(?, 193, 232) dtype=float32>>
#Out CNN1: <bound method Tensor.get_shape of <tf.Tensor 'model/batch_normalization_11/batchnorm/add_1:0' shape=(?, 187, 64) dtype=float32>>
#Out residual CNN2: <bound method Tensor.get_shape of <tf.Tensor 'model/Add:0' shape=(?, 187, 64) dtype=float32>>
#Out residual CNN3: <bound method Tensor.get_shape of <tf.Tensor 'model/Add_1:0' shape=(?, 187, 64) dtype=float32>>
#Max/Avg pooling
#Out max: <bound method Tensor.get_shape of <tf.Tensor 'model/Max:0' shape=(?, 64) dtype=float32>>
#Out avg: <bound method Tensor.get_shape of <tf.Tensor 'model/moments/Squeeze:0' shape=(?, 64) dtype=float32>>
#Out tmp_pool: <bound method Tensor.get_shape of <tf.Tensor 'model/concat_1:0' shape=(?, 128) dtype=float32>>
#flat_pool (?, 128)
#(?, 50)
#Number of parameters of the model: 223388

