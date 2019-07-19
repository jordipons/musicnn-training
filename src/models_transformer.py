import tensorflow as tf
import numpy as np


# TODO: multihead attention


def encoder(signal, is_training, config, channels, type):
    print('Transformer encoder')

    # Input feature map
    signal = tf.expand_dims(signal, 3)
    position = position_embedding_sin(signal.get_shape().as_list())
    embedding = tf.add(position,signal)
    norm_embedding = tf.contrib.layers.layer_norm(embedding)
    do_embedding = tf.nn.dropout(norm_embedding, 0.5)

    # layer_norm(x + self_attention(x))
    t1 = self_attention(embedding, channels)
    t1 = tf.expand_dims(t1, 3)
    res_t1 = tf.add(embedding, t1)
    norm_t1 = tf.contrib.layers.layer_norm(res_t1)
    do_t1 = tf.nn.dropout(norm_t1, 0.5)

    # layer_norm(x + convolution(x))
    ff = tf.layers.conv2d(inputs=do_t1,
                         filters=channels*3,
                         kernel_size=[1,channels],
                         padding="valid",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    ff_t = tf.transpose(ff, [0, 1, 3, 2]) 
    ff2 = tf.layers.conv2d(inputs=ff_t,
                         filters=channels,
                         kernel_size=[1,channels*3],
                         padding="valid",
                         activation=None,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    ff2_t = tf.transpose(ff2, [0, 1, 3, 2])  
    res_ff = tf.add(do_t1, ff2_t)
    norm_ff = tf.contrib.layers.layer_norm(res_ff)
    do_ff = tf.nn.dropout(norm_ff, 0.5)

    print('signal',signal.get_shape())
    print('position', position.get_shape())
    print('embedding', embedding.get_shape())
    print('t1', t1.get_shape())
    print('res_t1', res_t1.get_shape())
    print('do_t1', do_t1.get_shape())
    print('ff_t', ff_t.get_shape())
    print('res_ff', res_ff.get_shape())

    return tf.squeeze(do_ff, 3)


def self_attention(feature_map, channels):

    query = tf.layers.conv2d(inputs=feature_map,
                         filters=channels,
                         kernel_size=[1,feature_map.shape[2]],
                         padding="valid",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    key = tf.layers.conv2d(inputs=feature_map,
                         filters=channels,
                         kernel_size=[1,feature_map.shape[2]],
                         padding="valid",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    value = tf.layers.conv2d(inputs=feature_map,
                         filters=channels,
                         kernel_size=[1,feature_map.shape[2]],
                         padding="valid",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    query = tf.squeeze(query, 2)
    key = tf.squeeze(key, 2)
    value = tf.squeeze(value, 2)
    print('query', query.get_shape())
    print('key', key.get_shape())
    print('value', value.get_shape())
    # add dropout or layer norm? I don't think so

    logits = tf.matmul(query, key, transpose_b=True)
    #logits *= channels ** -0.5 ---------------------------------> do that!
    weights = tf.nn.softmax(logits, axis=1) # which axis?
    do_weights = tf.nn.dropout(weights, 0.5)
    attention_output = tf.matmul(do_weights, value)

    print('logits',logits.get_shape())
    return attention_output


def position_embedding(feature_map_size,is_training):
    _, T, num_units, _ = feature_map_size
    one_hot = tf.eye(T) # T vectors of T dimensions, each one enconding its position: identity matrix
    one_hot = tf.reshape(one_hot,[1,T,T,1])
    print('One hot position vector:',one_hot.get_shape())

    embed = tf.layers.conv2d(inputs=one_hot,
                            filters=num_units,
                            kernel_size=[1,T],
                            padding="valid",
                            activation=tf.nn.tanh, # Minz: Linear
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    return tf.reshape(embed,[-1, int(embed.shape[1]), int(embed.shape[3]), 1])


def position_embedding_sin(feature_map_size):
    _, T, num_units, _ = feature_map_size
    freq = 10000 # 10000 or 5
    print('freq:',freq)
    print('T:',T)
    print('num_units:',num_units)

    position_enc = np.array([[pos / np.power(freq, 2.*i/num_units) for i in range(num_units)] for pos in range(T)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    outputs = tf.convert_to_tensor(position_enc,dtype=tf.float32)
    #outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
    return tf.reshape(outputs,[-1,T,num_units,1])
