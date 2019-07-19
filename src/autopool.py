#!/usr/bin/env python
'''Autopool: Adaptive pooling operators for multiple instance learning'''

import tensorflow as tf

#from tf.keras import backend as K
#from tf.keras.engine.topology import Layer, InputSpec
#from tf.keras import initializers
#from tf.keras import constraints
#from tf.keras import regularizers


class AutoPool1D(tf.keras.layers.Layer):
    '''Automatically tuned soft-max pooling.

    This layer automatically adapts the pooling behavior to interpolate
    between mean- and max-pooling for each dimension.
    '''
    def __init__(self, axis=0,
                 kernel_initializer='zeros',
                 kernel_constraint=None,
                 kernel_regularizer=None,
                 **kwargs):
        '''

        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0 (should be time).

        kernel_initializer: Initializer for the weights matrix

        kernel_regularizer: Regularizer function applied to the weights matrix

        kernel_constraint: Constraint function applied to the weights matrix
        kwargs
        '''

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )

        super(AutoPool1D, self).__init__(**kwargs)

        self.axis = axis
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 3
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(1, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def get_config(self):
        config = {'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
                  'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
                  'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
                  'axis': self.axis}

        base_config = super(AutoPool1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        scaled = self.kernel * x
        max_val = tf.keras.backend.max(scaled, axis=self.axis, keepdims=True)
        softmax = tf.keras.backend.exp(scaled - max_val)
        weights = softmax / tf.keras.backend.sum(softmax, axis=self.axis, keepdims=True)
        return tf.keras.backend.sum(x * weights, axis=self.axis, keepdims=False)


class SoftMaxPool1D(tf.keras.layers.Layer):
    '''
    Keras softmax pooling layer.
    '''

    def __init__(self, axis=0, **kwargs):
        '''

        Parameters
        ----------
        axis : int
            Axis along which to perform the pooling. By default 0
            (should be time).
        kwargs
        '''
        super(SoftMaxPool1D, self).__init__(**kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        del shape[self.axis]
        return tuple(shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        max_val = tf.keras.backend.max(x, axis=self.axis, keepdims=True)
        softmax = tf.keras.backend.exp((x - max_val))
        weights = softmax / tf.keras.backend.sum(softmax, axis=self.axis, keepdims=True)
        return tf.keras.backend.sum(x * weights, axis=self.axis, keepdims=False)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SoftMaxPool1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
