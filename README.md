# NeuralArithmeticLogicUnits
Implementation of arithmetic logic units for neural networks in python (keras framework).

From [this](https://arxiv.org/abs/1808.00508 "Neural Arithmetic Logic Units") paper:

Neural Arithmetic Logic Unit (NALU), by analogy to the arithmetic logic unit in traditional processors. Experiments show that NALU-enhanced neural networks can learn to track time, perform arithmetic over images of numbers, translate numerical language into real-valued scalars, execute computer code, and count objects in images. In contrast to conventional architectures, we obtain substantially better generalization both inside and outside of the range of numerical values encountered during training, often extrapolating orders of magnitude beyond trained numerical ranges. 

Copy from a reddit users "BadGoyWithAGun" pastebin.

Copy from below or the NALU.py file in repository
 	

~~~~

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.models import *

class NALU(Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='M_hat')
        self.G = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.kernel_initializer,
                                 name='G')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W))
        g = K.sigmoid(K.dot(inputs, self.G))
        a = K.dot(x, W)
        output = g * a + (1 - g) * m
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    x = Input((10,))
    y = NALU(1)(x)
    m = Model(x, y)
    m.compile("adam", "mse")
    m.fit(np.random.rand(128, 10), np.random.rand(128, 1), 
          batch_size=128, epochs=2000)
          
~~~~
