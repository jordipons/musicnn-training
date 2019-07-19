import tensorflow as tf

def ismir_frontend(x, is_training, config, num_filt, type):

    print('input: ' + str(x.get_shape))
    expanded_layer = tf.expand_dims(x, 3)
    input_layer = tf.layers.batch_normalization(expanded_layer, training=is_training)
    print('input_layer: ', input_layer.get_shape())

    input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    input_pad_3 = tf.pad(input_layer, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")

    # TIMBRAL FEATURES
    f79 = timbral_block(inputs=input_pad_7,
                   filters=int(num_filt),
                   kernel_size=[7, int(0.9 * config['yInput'])],
                   is_training=is_training)

    f74 = timbral_block(inputs=input_pad_7,
                   filters=int(num_filt),
                   kernel_size=[7, int(0.4 * config['yInput'])],
                   is_training=is_training)

    f39 = timbral_block(inputs=input_pad_3,
                   filters=int(num_filt*2),
                   kernel_size=[3, int(0.9 * config['yInput'])],
                   is_training=is_training)

    f34 = timbral_block(inputs=input_pad_3,
                   filters=int(num_filt*2),
                   kernel_size=[3, int(0.4 * config['yInput'])],
                   is_training=is_training)

    f19 = timbral_block(inputs=input_layer,
                   filters=int(num_filt*4),
                   kernel_size=[1, int(0.9 * config['yInput'])],
                   is_training=is_training)

    f14 = timbral_block(inputs=input_layer,
                   filters=int(num_filt*4),
                   kernel_size=[1, int(0.4 * config['yInput'])],
                   is_training=is_training)

    # TEMPORAL FEATURES
    avg_input = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, config['yInput']],
                                        strides=[1, config['yInput']])
    avg_input_rs = tf.squeeze(avg_input, [3])

    t165 = tf.layers.conv1d(inputs=avg_input_rs,
                             filters=num_filt,
                             kernel_size=165,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_t165 = tf.layers.batch_normalization(t165, training=is_training)

    t128 = tf.layers.conv1d(inputs=avg_input_rs,
                             filters=num_filt*2,
                             kernel_size=128,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_t128 = tf.layers.batch_normalization(t128, training=is_training)

    t64 = tf.layers.conv1d(inputs=avg_input_rs,
                             filters=num_filt*4,
                             kernel_size=64,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_t64 = tf.layers.batch_normalization(t64, training=is_training)

    t32 = tf.layers.conv1d(inputs=avg_input_rs,
                              filters=num_filt*8,
                              kernel_size=32,
                              padding="same",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_t32 = tf.layers.batch_normalization(t32, training=is_training)

    # concatenate all feature maps
    pool = tf.concat([f79, f74, f39, f34, f19, f14, bn_t165, bn_t128, bn_t64, bn_t32], 2)
    print('Output pool: ', pool.get_shape())
    return pool

def frontend(x, is_training, config, num_filt, type):

    print('input: ' + str(x.get_shape))
    expanded_layer = tf.expand_dims(x, 3)
    input_layer = tf.layers.batch_normalization(expanded_layer, training=is_training)
    print('input_layer: ', input_layer.get_shape())


    if 'timbral' in type:

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        if '74' in type:
            f74 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.4 * config['yInput'])],
                           is_training=is_training)
        if '77' in type:
            f77 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.7 * config['yInput'])],
                           is_training=is_training)
        if 'v7' in type:
            v7 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(config['yInput'])],
                           is_training=is_training)


    if 'third' in type:

        input_layer = tf.layers.max_pooling2d(inputs=input_layer,
                           pool_size=[1, 3],
                           strides=[1, 3])
        print('reducing frequency resolution by three!')
        print(input_layer.get_shape())


    if 'temporal' in type:

        s1 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32), # if cannot reproduce results is because these were 128!
                          kernel_size=[128,1],
                          is_training=is_training)

        s2 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[64,1],
                          is_training=is_training)

        s3 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[32,1],
                          is_training=is_training)


    if 'invariant' in type:

        ti = tempo_invariant_block(inputs=input_layer,
                                   filters=int(num_filt*32),
                                   kernel_size=[3,1], # 64 frames = 1 sec that will be dilated
                                   dilations=[[2,1],[4,1],[6,1]],
                                   name='tempo_invariant_short',
                                   is_training=is_training)

        tii = tempo_invariant_block(inputs=input_layer,
                                   filters=int(num_filt*32),
                                   kernel_size=[32,1], # 64 frames = 1 sec that will be dilated
                                   dilations=[[2,1],[3,1],[4,1],[5,1],[6,1]],
                                   name='tempo_invariant_long',
                                   is_training=is_training)

    print(type) 
    # choose the feature maps we want to use for the experiment
    if type == '7774timbraltemporalthird' or type == '7774timbraltemporal':
        concat_list = [f74, f77, s1, s2, s3]

    elif type == '7774timbralinvariantthird' or type == '7774timbralinvariant':
        concat_list = [f74, f77, ti, tii]

    elif type == 'v77774timbraltemporalthird' or type == 'v77774timbraltemporal':
        concat_list = [f74, f77, s1, s2, s3, v7]

    elif type == '7774timbral':
        concat_list = [f74, f77]

    elif type == '74timbral':
        concat_list = [f74]

    elif type == '77timbral':
        concat_list = [f77]

    return tf.concat(concat_list, 2)
    #return tf.expand_dims(concat_front_end, 3)


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


def tempo_block(inputs, filters, kernel_size, is_training, padding="same", activation=tf.nn.relu,
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
    print('Output tempo-variant block: ', out.get_shape())
    return out


def tempo_invariant_block(inputs, filters, kernel_size, dilations, name, is_training, padding="same",
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
    feature_maps = []

    # create base filter
    with tf.variable_scope(name):
        feature_maps.append(tf.layers.conv2d(inputs=inputs,
                                             filters=filters,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             activation=activation,
                                             kernel_initializer=kernel_initializer,
                                             name='shared_weights'))

    # reuse the weights to create tempor invariant filters with dilated filters
    for d in dilations:
        with tf.variable_scope(name, reuse=True):
            feature_maps.append(tf.layers.conv2d(inputs=inputs,
                                                 filters=filters,
                                                 kernel_size=kernel_size,
                                                 dilation_rate=d,
                                                 padding=padding,
                                                 activation=activation,
                                                 kernel_initializer=kernel_initializer,
                                                 name='shared_weights'))
    # concatenate all feature maps
    for f in feature_maps:
        print(f.get_shape())
    concat_front_end = tf.concat(feature_maps, 2)

    bn = tf.layers.batch_normalization(concat_front_end, training=is_training)
    print(bn.get_shape())
    print([v.get_shape().as_list() for v in tf.trainable_variables()])
    pool = tf.layers.max_pooling2d(inputs=bn,
                                   pool_size=[1, bn.shape[2]],
                                   strides=[1, bn.shape[2]])
    out = tf.squeeze(pool, [2])
    print('Output tempo-variant block: ', out.get_shape())
    return out

