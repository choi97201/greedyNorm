from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.layers import Lambda

import numpy as np
import tensorflow as tf
session = tf.Session()
session.run(tf.initialize_all_variables())

from keras.utils.generic_utils import get_custom_objects

def linear_transform(x):
    v1 = tf.constant(2., name='multiplier')
    v2 = tf.Variable(1., name='bias')
    # tmp = tf.math.divide(input, 64)
    # splits = tf.split(tmp[0].shape,64,x)
    # for i in range(len(splits)):
    #     splits[i]*v1
    # x = tf.concat(0,splits)
    # return splits
    #v1 = np.ones()
    #x = tf.matmul(x,v1)
    #print("=================in function==============")
    input_shape = K.int_shape(x)
    temp = np.ones(input_shape[0],dtype=np.float32)
    for i in range(len(temp)):
        temp[i] = float(i)
    #print("====================before temp==================")
    #print(temp)
    temp = tf.convert_to_tensor(temp, dtype=tf.float32)
    temp = K.reshape(temp, input_shape)
    #print("====================after temp==================")
    #print(temp)
    x = temp
    #print(input_shape[0])
    return x

class GroupNormalization(Layer):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=1,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)

            # #print("gamma=", self.gamma)
            # #print("gamma shape=", self.gamma.shape)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        #print("1. input_shape",input_shape)  #input_shape (None, 512, 512, 64)
        tensor_input_shape = K.shape(inputs)
        #print("2. tensor_input_shape",tensor_input_shape) #tensor_input_shape Tensor("group_normalization_1/Shape:0", shape=(4,), dtype=int32)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        #print("3. broadcast_shape",broadcast_shape) #broadcast_shape [1, 1, 1, 1]
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        #print("4. reshape_group_shape",reshape_group_shape) #reshape_group_shape Tensor("group_normalization_1/Shape_1:0", shape=(4,), dtype=int32)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        #print("5. group_shape",group_shape) #group_shape Tensor("group_normalization_1/stack:0", shape=(5,), dtype=int32)
        inputs = K.reshape(inputs, group_shape)
        #print("6. inputs",inputs) #inputs Tensor("group_normalization_1/Reshape:0", shape=(?, 32, ?, ?, 2), dtype=float32)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma
            #print("broadcast_gamma.shape=",broadcast_gamma.shape)
        name = outputs.name.split("/")[0]
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)
        scale_shape = K.int_shape(variables[0])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            temp = np.ones(scale_shape[0], dtype=np.float32)
            for i in range(len(temp)): #scale 바꿔주는 부분
                temp[i] = float(i)
            temp = tf.convert_to_tensor(temp, dtype=tf.float32)
            temp = K.reshape(temp, scale_shape)
            #print(temp.shape)
            #print(sess.run(variables[0]))

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta
            #print("broadcast_beta.shape=",broadcast_beta.shape)

        outputs = K.reshape(outputs, tensor_input_shape)
        #outputs = outputs * temp
        #print("============here===========")
        #print("7. outputs",outputs) #outputs Tensor("group_normalization_1/Reshape_4:0", shape=(?, 512, 512, 64), dtype=float32)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    ip = Input(shape=(None, None, 4))
    #ip = Input(batch_shape=(100, None, None, 2))
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1)(ip)
    model = Model(ip, x)
    model.summary()

