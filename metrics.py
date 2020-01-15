from functools import wraps
import tensorflow as tf
import keras.backend as K

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def preprocess(y_true, y_pred):
    s = K.int_shape(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    y_pred = K.one_hot(K.argmax(y_pred), s[-1])

    return y_true, y_pred, s

# def pix_acc(y_true, y_pred):
#     y_true, y_pred, _ = preprocess(y_true, y_pred)
    
#     correct = K.cast(K.equal(y_pred, y_true), dtype='float32')
#     acc = K.sum(correct) / K.cast(K.prod(K.shape(y_pred)), dtype='float32')
    
#     return acc


# def mean_acc1(y_true, y_pred):
#     y_true, y_pred, _ = preprocess(y_true, y_pred)
	
#     equal = K.cast(K.equal(y_pred, y_true), dtype='float32') * y_true
#     correct = K.sum(equal, axis=1)
#     pixels_per_class = K.sum(y_true, axis=1)

#     # acc = correct / (pixels_per_class + K.epsilon())
#     acc = correct / pixels_per_class
#     acc_mask = tf.is_finite(acc)
#     acc_masked = tf.boolean_mask(acc, acc_mask)
    
#     return K.mean(acc_masked)


# def iou(y_true, y_pred):
#     y_true, y_pred, _ = preprocess(y_true, y_pred)
    
#     equal = K.cast(K.equal(y_pred, y_true), dtype='float32') * y_true
#     intersection = K.sum(equal, axis=1)
#     union_per_class = K.sum(y_true, axis=1) + K.sum(y_pred, axis=1)

#     # iou = intersection / (union_per_class - intersection + K.epsilon())
#     iou = intersection / (union_per_class - intersection)
#     iou_mask = tf.is_finite(iou)
#     iou_masked = tf.boolean_mask(iou, iou_mask)

#     return K.mean(iou_masked)

# def tp_func(y_true, y_pred):
#     return K.sum(K.cast(y_true * y_pred, dtype='float32'), axis=1)

# def fp_func(y_true, y_pred):
#     return K.sum(K.cast((1-y_true) * y_pred, dtype='float32'), axis=1)

# def fn_func(y_true, y_pred):
#     return K.sum(K.cast(y_true * (1-y_pred), dtype='float32'), axis=1)

# def pre(y_true, y_pred):
#     y_true, y_pred, _ = preprocess(y_true, y_pred)

#     tp = tp_func(y_true, y_pred)
#     fp = fp_func(y_true, y_pred)
#     precision = tp / (tp + fp)

#     precision_mask = tf.is_finite(precision)
#     precision_masked = tf.boolean_mask(precision, precision_mask)

#     return K.mean(precision_masked)

# def re(y_true, y_pred):
#     y_true, y_pred, _ = preprocess(y_true, y_pred)

#     tp = tp_func(y_true, y_pred)
#     fn = fn_func(y_true, y_pred)
#     recall = tp / (tp + fn)

#     recall_mask = tf.is_finite(recall)
#     recall_masked = tf.boolean_mask(recall, recall_mask)

#     return K.mean(recall_masked)

# def f1(y_true, y_pred):
#     precision = pre(y_true, y_pred)
#     recall = re(y_true, y_pred)
#     f1 = 2 *precision * recall / (precision + recall)

#     return K.mean(f1)

# def DSC(y_true, y_pred):
#     y_true, y_pred, _ = preprocess(y_true, y_pred)

#     intersection = tp_func(y_true, y_pred)
#     union = K.sum(y_true, axis=1) + K.sum(y_pred, axis=1)
#     dice = 2 * intersection / union

#     dice_mask = tf.is_finite(dice)
#     dice_masked = tf.boolean_mask(dice, dice_mask)

#     return K.mean(dice_masked)

    

def as_keras_metric(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        value, update_op = method(self, *args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def mean_acc(y_true, y_pred):
    y_true, y_pred, _ = preprocess(y_true, y_pred)
    return tf.metrics.accuracy(y_true, y_pred)

@as_keras_metric
def iou(y_true, y_pred):
    y_true, y_pred, s = preprocess(y_true, y_pred)
    return tf.metrics.mean_iou(y_true, y_pred, s[-1])


@as_keras_metric
def pre(y_true, y_pred):
    y_true, y_pred, _ = preprocess(y_true, y_pred)
    return tf.metrics.precision(y_true, y_pred)

@as_keras_metric
def re(y_true, y_pred):
    y_true, y_pred, _ = preprocess(y_true, y_pred)
    return tf.metrics.recall(y_true, y_pred)

def f1(y_true, y_pred):
    _recall = re(y_true, y_pred)
    _precision = pre(y_true, y_pred)
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())
    return _f1score

@as_keras_metric
def true_positives(y_true, y_pred):
    return tf.metrics.true_positives(y_true, y_pred)

@as_keras_metric
def false_positives(y_true, y_pred):
    return tf.metrics.false_positives(y_true, y_pred)

@as_keras_metric
def false_negatives(y_true, y_pred):
    return tf.metrics.false_negatives(y_true, y_pred)

def DSC(y_true, y_pred):
    y_true, y_pred, _ = preprocess(y_true, y_pred)
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    dice_score = K.mean(2 * tp / (2 * tp + fp + fn + K.epsilon()))
    return dice_score

def mIOU(y_true,y_pred):
    y_true,y_pred, _ = preprocess(y_true,y_pred)

def Pixelacc(y_true,y_pred):
    pass

def Per_clas_sacc(y_true,y_pred):
    pass

