import tensorflow as tf
from tensorflow.keras import backend as K

SMOOTH = 1e-5


# ----------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------

def _gather_channels(x, indexes):
    if indexes is None:
        return x
    if isinstance(indexes, int):
        indexes = [indexes]
    if K.image_data_format() == 'channels_last':
        x = tf.transpose(x, (4, 0, 1, 2, 3))
        x = tf.gather(x, indexes, axis=0)
        x = tf.transpose(x, (1, 2, 3, 4, 0))
    else:
        x = tf.transpose(x, (1, 0, 2, 3, 4))
        x = tf.gather(x, indexes, axis=0)
        x = tf.transpose(x, (1, 0, 2, 3, 4))
    return x


def get_reduce_axes(per_image):
    axes = [1, 2, 3] if K.image_data_format() == 'channels_last' else [2, 3, 4]
    if not per_image:
        axes.insert(0, 0)
    return axes


def round_if_needed(x, threshold):
    if threshold is not None:
        x = tf.cast(x > threshold, K.floatx())
    return x


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = tf.reduce_mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return tf.reduce_mean(x)


# ----------------------------------------------------------------
#   Metric / Loss functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = _gather_channels(gt, class_indexes), _gather_channels(pr, class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    intersection = tf.reduce_sum(gt * pr, axis=axes)
    union = tf.reduce_sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    return average(score, per_image, class_weights)


def f_score(gt, pr, beta=1, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = _gather_channels(gt, class_indexes), _gather_channels(pr, class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    fn = tf.reduce_sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return average(score, per_image, class_weights)


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = _gather_channels(gt, class_indexes), _gather_channels(pr, class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)
    return average(score, per_image, class_weights)


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = _gather_channels(gt, class_indexes), _gather_channels(pr, class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    tp = tf.reduce_sum(gt * pr, axis=axes)
    fn = tf.reduce_sum(gt, axis=axes) - tp
    score = (tp + smooth) / (tp + fn + smooth)
    return average(score, per_image, class_weights)


def binary_crossentropy(gt, pr):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt, pr))


def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None):
    gt, pr = _gather_channels(gt, class_indexes), _gather_channels(pr, class_indexes)
    axis = 4 if K.image_data_format() == 'channels_last' else 1
    pr /= tf.reduce_sum(pr, axis=axis, keepdims=True)
    pr = tf.clip_by_value(pr, K.epsilon(), 1 - K.epsilon())
    loss = -gt * tf.math.log(pr) * class_weights
    return tf.reduce_mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25):
    pr = tf.clip_by_value(pr, K.epsilon(), 1. - K.epsilon())
    loss_1 = -gt * (alpha * tf.pow(1 - pr, gamma) * tf.math.log(pr))
    loss_0 = -(1 - gt) * ((1 - alpha) * tf.pow(pr, gamma) * tf.math.log(1 - pr))
    return tf.reduce_mean(loss_0 + loss_1)


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None):
    gt, pr = _gather_channels(gt, class_indexes), _gather_channels(pr, class_indexes)
    pr = tf.clip_by_value(pr, K.epsilon(), 1. - K.epsilon())
    loss = -gt * (alpha * tf.pow(1 - pr, gamma) * tf.math.log(pr))
    return tf.reduce_mean(loss)


# ----------------------------------------------------------------
#   Keras Loss Wrappers
# ----------------------------------------------------------------

class JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return 1 - iou_score(y_true, y_pred,
                              class_weights=self.class_weights,
                              class_indexes=self.class_indexes,
                              per_image=self.per_image,
                              smooth=self.smooth)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return 1 - f_score(y_true, y_pred,
                           beta=self.beta,
                           class_weights=self.class_weights,
                           class_indexes=self.class_indexes,
                           per_image=self.per_image,
                           smooth=self.smooth)


class BinaryCELoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name='binary_crossentropy')

    def call(self, y_true, y_pred):
        return binary_crossentropy(y_true, y_pred)


class CategoricalCELoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes

    def call(self, y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred,
                                       class_weights=self.class_weights,
                                       class_indexes=self.class_indexes)


class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        return binary_focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, class_indexes=None):
        super().__init__(name='categorical_focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_indexes = class_indexes

    def call(self, y_true, y_pred):
        return categorical_focal_loss(y_true, y_pred,
                                      alpha=self.alpha,
                                      gamma=self.gamma,
                                      class_indexes=self.class_indexes)


# ----------------------------------------------------------------
#   Aliases
# ----------------------------------------------------------------

jaccard_loss = JaccardLoss()
dice_loss = DiceLoss()
binary_crossentropy = BinaryCELoss()
categorical_crossentropy = CategoricalCELoss()
binary_focal_loss = BinaryFocalLoss()
categorical_focal_loss = CategoricalFocalLoss()
