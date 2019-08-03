import tensorflow as tf

def dieleman(x, is_training, config):
    print('Input: ' + str(x.get_shape))
    input_layer = tf.expand_dims(x, 3)
    bn_input = tf.compat.v1.layers.batch_normalization(input_layer, training=is_training)

    conv1 = tf.compat.v1.layers.conv2d(inputs=bn_input, 
    	                     filters=32, 
    	                     kernel_size=[8, config['yInput']], 
    	                     padding="valid", 
    	                     activation=tf.nn.relu, 
    	                     name='1cnnOut', 
    	                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[4, 1], strides=[4, 1], name='1-pool')
    pool1_rs = tf.reshape(pool1, [-1, int(pool1.shape[1]), int(pool1.shape[3]), 1])
    print('\t\t' + str(pool1_rs.get_shape))

    conv2 = tf.compat.v1.layers.conv2d(inputs=pool1_rs, 
    	                     filters=32, 
    	                     kernel_size=[8, pool1_rs.shape[2]], 
    	                     padding="valid", 
    	                     activation=tf.nn.relu, 
    	                     name='2cnnOut', 
    	                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[4, 1], strides=[4, 1], name='2-pool')
    flat_pool2 = tf.reshape(pool2,[-1,int(pool2.shape[1]*pool2.shape[2]*pool2.shape[3])]) # flatten
    print('\t\t' + str(flat_pool2.shape))

    dense = tf.compat.v1.layers.dense(inputs=flat_pool2, 
    	                    activation=tf.nn.relu, 
    	                    units=100, 
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    output = tf.compat.v1.layers.dense(inputs=dense, 
    	                   activation=None, 
    	                   units=config['num_classes_dataset'], 
    	                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    print('output: ' + str(output.get_shape))
    return output 
    
    
def vgg(x, is_training, config, num_filters=32):
    print('Input: ' + str(x.get_shape))
    input_layer = tf.expand_dims(x, 3)
    bn_input = tf.compat.v1.layers.batch_normalization(input_layer, training=is_training)

    conv1 = tf.compat.v1.layers.conv2d(inputs=bn_input,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='1CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training)
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv1, pool_size=[4, 1], strides=[2, 2])
    print('pool1: ' + str(pool1.get_shape))

    do_pool1 = tf.compat.v1.layers.dropout(pool1, rate=0.25, training=is_training)
    conv2 = tf.compat.v1.layers.conv2d(inputs=do_pool1,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='2CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training)
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv2, pool_size=[2, 2], strides=[2, 2])
    print('pool2: ' + str(pool2.get_shape))

    do_pool2 = tf.compat.v1.layers.dropout(pool2, rate=0.25, training=is_training)
    conv3 = tf.compat.v1.layers.conv2d(inputs=do_pool2,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='3CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training)
    pool3 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv3, pool_size=[2, 2], strides=[2, 2])
    print('pool3: ' + str(pool3.get_shape))

    do_pool3 = tf.layers.dropout(pool3, rate=0.25, training=is_training)
    conv4 = tf.layers.conv2d(inputs=do_pool3,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding='same',
                             activation=tf.nn.relu,
                             name='4CNN',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv4 = tf.compat.v1.layers.batch_normalization(conv4, training=is_training)
    pool4 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv4, pool_size=[2, 2], strides=[2, 2])
    print('pool4: ' + str(pool4.get_shape))

    do_pool4 = tf.compat.v1.layers.dropout(pool4, rate=0.25, training=is_training)
    conv5 = tf.compat.v1.layers.conv2d(inputs=do_pool4, 
                             filters=num_filters, 
                             kernel_size=[3, 3], 
                             padding='same', 
                             activation=tf.nn.relu,
                             name='5CNN', 
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv5 = tf.compat.v1.layers.batch_normalization(conv5, training=is_training)
    pool5 = tf.layers.max_pooling2d(inputs=bn_conv5, pool_size=[4, 4], strides=[4, 4])
    print('pool5: ' + str(pool5.get_shape))

    flat_pool5 = tf.contrib.layers.flatten(pool5)
    do_pool5 = tf.compat.v1.layers.dropout(flat_pool5, rate=0.5, training=is_training)
    output = tf.compat.v1.layers.dense(inputs=do_pool5,
                            activation=None,
                            units=config['num_classes_dataset'],
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    print('output: ' + str(output.get_shape))    
    return output    


def timbre(x, is_training, config, num_filt=1):
    print('Input: ' + str(x.get_shape))
    expanded_layer = tf.expand_dims(x, 3)
    input_layer = tf.compat.v1.layers.batch_normalization(expanded_layer, training=is_training)
    
    # FRONT END
    
    # padding only time domain for an efficient 'same' implementation
    # (since we pool throughout all frequency afterwards)
    input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    input_pad_5 = tf.pad(input_layer, [[0, 0], [2, 2], [0, 0], [0, 0]], "CONSTANT")    
    input_pad_3 = tf.pad(input_layer, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")

    # [TIMBRE] filter shape 1: 7x0.8f
    conv1 = tf.compat.v1.layers.conv2d(inputs=input_pad_7,
                             filters=3*num_filt,
                             kernel_size=[7, int(0.8 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training)
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv1,
                                    pool_size=[1, bn_conv1.shape[2]],
                                    strides=[1, bn_conv1.shape[2]])
    p1 = tf.squeeze(pool1, [2])

    # [TIMBRE] filter shape 2: 5x0.8f
    conv2 = tf.compat.v1.layers.conv2d(inputs=input_pad_5,
                             filters=3*num_filt,
                             kernel_size=[5, int(0.8 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training)
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv2,
                                    pool_size=[1, bn_conv2.shape[2]],
                                    strides=[1, bn_conv2.shape[2]])
    p2 = tf.squeeze(pool2, [2])    

    # [TIMBRE] filter shape 3: 3x0.8f
    conv3 = tf.compat.v1.layers.conv2d(inputs=input_pad_3, 
                             filters=6*num_filt,
                             kernel_size=[3, int(0.8 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training)
    pool3 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv3,
                                    pool_size=[1, bn_conv3.shape[2]],
                                    strides=[1, bn_conv3.shape[2]])
    p3 = tf.squeeze(pool3, [2])

    # [TIMBRE] filter shape 4: 1x0.8f
    conv4 = tf.compat.v1.layers.conv2d(inputs=input_layer, 
                             filters=10*num_filt,
                             kernel_size=[1, int(0.8 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv4 = tf.compat.v1.layers.batch_normalization(conv4, training=is_training)
    pool4 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv4,
                                    pool_size=[1, bn_conv4.shape[2]],
                                    strides=[1, bn_conv4.shape[2]])
    p4 = tf.squeeze(pool4, [2])

    # [TIMBRE] filter shape 5: 7x0.6f
    conv5 = tf.compat.v1.layers.conv2d(inputs=input_pad_7,
                             filters=5*num_filt,
                             kernel_size=[7, int(0.6 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv5 = tf.compat.v1.layers.batch_normalization(conv5, training=is_training)
    pool5 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv5,
                                    pool_size=[1, bn_conv5.shape[2]],
                                    strides=[1, bn_conv5.shape[2]])
    p5 = tf.squeeze(pool5, [2])

    # [TIMBRE] filter shape 6: 5x0.6f
    conv6 = tf.compat.v1.layers.conv2d(inputs=input_pad_5, 
                             filters=5*num_filt,
                             kernel_size=[5, int(0.6 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv6 = tf.compat.v1.layers.batch_normalization(conv6, training=is_training)
    pool6 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv6, 
    	                            pool_size=[1, bn_conv6.shape[2]],
                                    strides=[1, bn_conv6.shape[2]])
    p6 = tf.squeeze(pool6, [2])    

    # [TIMBRE] filter shape 7: 3x0.6f
    conv7 = tf.compat.v1.layers.conv2d(inputs=input_pad_3, 
                             filters=10*num_filt,
                             kernel_size=[3, int(0.6 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv7 = tf.compat.v1.layers.batch_normalization(conv7, training=is_training)
    pool7 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv7, 
    	                            pool_size=[1, bn_conv7.shape[2]],
                                    strides=[1, bn_conv7.shape[2]])
    p7 = tf.squeeze(pool7, [2])

    # [TIMBRE] filter shape 8: 1x0.6f
    conv8 = tf.compat.v1.layers.conv2d(inputs=input_layer, 
                             filters=15*num_filt,
                             kernel_size=[1, int(0.6 * config['yInput'])], padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv8 = tf.compat.v1.layers.batch_normalization(conv8, training=is_training)
    pool8 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv8, 
    	                            pool_size=[1, bn_conv8.shape[2]],
                                    strides=[1, bn_conv8.shape[2]])
    p8 = tf.squeeze(pool8, [2])

    # [TIMBRE] filter shape 9: 7x0.2f
    conv9 = tf.compat.v1.layers.conv2d(inputs=input_pad_7,
                             filters=5*num_filt,
                             kernel_size=[7, int(0.2 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv9 = tf.compat.v1.layers.batch_normalization(conv9, training=is_training)
    pool9 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv9,
                                    pool_size=[1, bn_conv9.shape[2]],
                                    strides=[1, bn_conv9.shape[2]])
    p9 = tf.squeeze(pool9, [2])

    # [TIMBRE] filter shape 10: 5x0.2f
    conv10 = tf.compat.v1.layers.conv2d(inputs=input_pad_5, 
                             filters=5*num_filt,
                             kernel_size=[5, int(0.2 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv10 = tf.compat.v1.layers.batch_normalization(conv10, training=is_training)
    pool10 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv10, 
    	                            pool_size=[1, bn_conv10.shape[2]],
                                    strides=[1, bn_conv10.shape[2]])
    p10 = tf.squeeze(pool10, [2])    

    # [TIMBRE] filter shape 11: 3x0.2f
    conv11 = tf.compat.v1.layers.conv2d(inputs=input_pad_3, 
                             filters=10*num_filt,
                             kernel_size=[3, int(0.2 * config['yInput'])],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv11 = tf.compat.v1.layers.batch_normalization(conv11, training=is_training)
    pool11 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv11, 
    	                            pool_size=[1, bn_conv11.shape[2]],
                                    strides=[1, bn_conv11.shape[2]])
    p11 = tf.squeeze(pool11, [2])

    # [TIMBRE] filter shape 12: 1x0.2f
    conv12 = tf.compat.v1.layers.conv2d(inputs=input_layer, 
                             filters=15*num_filt,
                             kernel_size=[1, int(0.2 * config['yInput'])], padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv12 = tf.compat.v1.layers.batch_normalization(conv12, training=is_training)
    pool12 = tf.compat.v1.layers.max_pooling2d(inputs=bn_conv12, 
    	                            pool_size=[1, bn_conv12.shape[2]],
                                    strides=[1, bn_conv12.shape[2]])
    p12 = tf.squeeze(pool12, [2])

    # concatenate all feature maps
    pool = tf.concat([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], 2)
    out_front_end =  tf.expand_dims(pool, 3)    
    
    # BACK END
    conv2 = tf.compat.v1.layers.conv2d(inputs=out_front_end, 
    	                     filters=32, 
    	                     kernel_size=[8, out_front_end.shape[2]], 
    	                     padding="valid", 
    	                     activation=tf.nn.relu, 
    	                     name='2cnnOut', 
    	                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    print(conv2.get_shape)
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, 
    	                            pool_size=[4, 1], 
    	                            strides=[4, 1], 
    	                            name='2-pool')
    print(pool2.get_shape)
    flat_pool2 = tf.reshape(pool2,[-1,int(pool2.shape[1]*pool2.shape[2]*pool2.shape[3])]) # flatten
    print(flat_pool2.shape)
    dense = tf.compat.v1.layers.dense(inputs=flat_pool2, 
    	                    activation=tf.nn.relu, 
    	                    units=100, 
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    output = tf.compat.v1.layers.dense(inputs=dense, 
    	                     activation=None, 
    	                     units=config['num_classes_dataset'], 
    	                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return output

