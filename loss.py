import keras
import tensorflow as tf
import keras.backend as K

_EPSILON = 1e-7

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def epsilon():
    return _EPSILON

def categorical_crossentropy(l1=0., l2=0.):
    def crossentropy(y_true, y_pred):
        loss = K.categorical_crossentropy(y_true, y_pred) + l1 * K.mean(K.abs(y_true-y_pred)) + l2 * K.mean(K.square(y_true-y_pred))
        return loss
    return crossentropy

def categorical_hinge(l1=0., l2=0.):
    def hinge(y_true, y_pred, axis=-1):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1) + l1 * K.mean(K.abs(y_true - y_pred)) + l2 * K.mean(K.square(y_true - y_pred))
        return loss
    return hinge

def categorical_hinge_expotential_regularization(l1=0.,l2=0.):
     def hinge(y_true, y_pred, axis=-1):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1) + l1 * K.mean(K.abs(y_true - y_pred))*tf.math.exp(l2 * K.mean(K.square(y_true - y_pred)))
        return loss
     return hinge

# 제안하는 focal함수를 이용하는 hinge 함수
#def categorical_hinge_focal(gamma=2., alpha=.25, l1=0., l2=0.):
#    def hinge(y_true, y_pred, axis=-1):
#        output_dimensions = list(range(len(y_pred.get_shape())))
#        if axis != -1 and axis not in output_dimensions:
#            raise ValueError('{}{}{}'.format('Unexpected channels axis {}. '.format(axis),
#                                             'Expected to be -1 or one of the axes of `output`, ',
#                                             'which has {} dimensions.'.format(len(y_pred.get_shape()))))
#        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
#        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
#        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
#        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
#        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
#        pos = K.sum(y_true * y_pred, axis=-1)
#        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
#       loss = -tf.reduce_sum(alpha * tf.pow(1 - K.mean(K.maximum(0.0, neg - pos + 1), axis=-1), gamma) * y_true * tf.log(
#           K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)), axis) + l1 * K.mean(
#           K.abs(y_true - y_pred),axis) + l2 * K.mean(K.square(y_true - y_pred),axis)
#        return loss
#    return hinge
def categorical_hinge_focal(gamma=2., alpha=.25, l1=0., l2=0.):
    def hinge_focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError('{}{}{}'.format('Unexpected channels axis {}. '.format(axis),
                                             'Expected to be -1 or one of the axes of `output`, ',
                                             'which has {} dimensions.'.format(len(y_pred.get_shape()))))
        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        loss = -tf.reduce_sum(alpha * tf.pow(1 - K.mean(K.maximum(0.0, neg - pos + 1), axis=-1), gamma) * y_true * tf.log(
                K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)), axis) + l1 * K.mean(
            K.abs(y_true - y_pred),axis) + l2 * K.mean(K.square(y_true - y_pred))
        return loss
    return hinge_focal


def categorical_hinge_focal_expotential_regularization(gamma=2., alpha=.25, l1=0., l2=0.):
    def hinge_focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError('{}{}{}'.format('Unexpected channels axis {}. '.format(axis),
                                             'Expected to be -1 or one of the axes of `output`, ',
                                             'which has {} dimensions.'.format(len(y_pred.get_shape()))))
        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        loss = -tf.reduce_sum(alpha * tf.pow(1 - K.mean(K.maximum(0.0, neg - pos + 1), axis=-1), gamma) * y_true * tf.log(K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)), axis) + l1 * K.mean(K.abs(y_true - y_pred))*tf.math.exp(l2 * K.mean(K.square(y_true - y_pred)))
        return loss
    return hinge_focal

def categorical_crossentropy_expotential_regularization(l1=0., l2=0.):
    def crossentropy(y_true, y_pred):
        loss = K.categorical_crossentropy(y_true, y_pred) + l1 * K.mean(K.abs(y_true - y_pred))*tf.math.exp(l2 * K.mean(K.square(y_true - y_pred)))
        return loss
    return crossentropy

def categorical_crossentropy_focal(gamma=2., alpha=.25, l1=0., l2=0.):
    def crossentropy_focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))
        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        loss = -tf.reduce_sum(alpha * tf.pow(1 - K.categorical_crossentropy(y_true, _y_pred), gamma) * y_true * tf.log(K.categorical_crossentropy(y_true, _y_pred)), axis) + l1 * K.mean(K.abs(y_true - y_pred))+l2 * K.mean(K.square(y_true - y_pred))
        return loss
    return crossentropy_focal

def categorical_crossentropy_focal_expotential_regularization(gamma=2., alpha=.25, l1=0., l2=0.):
    def crossentropy_focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))
        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        loss = -tf.reduce_sum(alpha * tf.pow(1 - K.categorical_crossentropy(y_true, _y_pred), gamma) * y_true * tf.log(K.categorical_crossentropy(y_true, _y_pred)), axis) + l1 * K.mean(K.abs(y_true - y_pred))*tf.math.exp(l2 * K.mean(K.square(y_true - y_pred)))
        return loss
    return crossentropy_focal

def categorical_focal(gamma=2., alpha=.25, l1=0., l2=0.):
    def focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))

        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        loss = -tf.reduce_sum(alpha * tf.pow(1-_y_pred, gamma) * y_true * tf.log(_y_pred), axis) + l1 * K.mean(K.abs(y_true-y_pred)) + l2 * K.mean(K.square(y_true-y_pred))
        return loss
    return focal

def categorical_focal_expotential_regularization(gamma=2., alpha=.25, l1=0., l2=0.):
    def focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))

        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        loss = -tf.reduce_sum(alpha * tf.pow(1-_y_pred, gamma) * y_true * tf.log(_y_pred), axis) + l1 * K.mean(K.abs(y_true-y_pred))*tf.math.exp(l2 * K.mean(K.square(y_true-y_pred)))
        return loss
    return focal

def categorical_focal_focal(gamma=2., alpha=.25, l1=0., l2=0.):
    def focal_focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))
        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        _y_pred = alpha * tf.pow(1 - _y_pred, gamma) * y_true * tf.log(_y_pred) + alpha * tf.pow(_y_pred, gamma) * (
                    1 - y_true) * tf.log(1 - _y_pred)
        loss = alpha * tf.pow(1 - _y_pred, gamma) * y_true * tf.log(_y_pred) + alpha * tf.pow(_y_pred, gamma) * (
                    1 - y_true) * tf.log(1 - _y_pred)
        loss = -tf.reduce_sum(loss, axis) + l1 * K.mean(K.abs(y_true - y_pred)) + l2 * K.mean(K.square(y_true - y_pred))
        return loss
    return focal_focal

def categorical_focal_focal_expotential_regularization(gamma=2., alpha=.25, l1=0., l2=0.):
    def focal_focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))
        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _alpha = _to_tensor(alpha, _y_pred.dtype.base_dtype)
        _gamma = _to_tensor(gamma, _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        _y_pred = alpha * tf.pow(1 - _y_pred, gamma) * y_true * tf.log(_y_pred) + alpha * tf.pow(_y_pred, gamma) * ( 1 - y_true) * tf.log(1 - _y_pred)
        loss = alpha * tf.pow(1 - _y_pred, gamma) * y_true * tf.log(_y_pred) + alpha * tf.pow(_y_pred, gamma) * ( 1 - y_true) * tf.log(1 - _y_pred)
        loss = -tf.reduce_sum(loss, axis) + l1 * K.mean(K.abs(y_true - y_pred))* tf.math.exp(l2 * K.mean(K.square(y_true - y_pred)))
        return loss
    return focal_focal

def binary_crossentropy(l1=0., l2=0.):
    def crossentropy(y_true, y_pred):
        loss = K.binary_crossentropy(y_true, y_pred) + l1 * K.mean(K.abs(y_true-y_pred)) + l2 * K.mean(K.square(y_true-y_pred))
        return loss
    return crossentropy


def binary_focal(gamma=2., alpha=.25, l1=0., l2=0.):
    def focal(y_true, y_pred, axis=-1):
        output_dimensions = list(range(len(y_pred.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(y_pred.get_shape()))))

        _y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(epsilon(), _y_pred.dtype.base_dtype)
        _y_pred = tf.clip_by_value(_y_pred, _epsilon, 1. - _epsilon)
        loss = alpha * tf.pow(1-_y_pred, gamma) * y_true * tf.log(_y_pred) + alpha * tf.pow(_y_pred, gamma) * (1-y_true) * tf.log(1-_y_pred)
        loss = -tf.reduce_sum(loss, axis) + l1 * K.mean(K.abs(y_true-y_pred)) + l2 * K.mean(K.square(y_true-y_pred))
        return loss
    return focal
