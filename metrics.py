import tensorflow as tf
from tensorflow.keras import backend as K

SMOOTH = 1e-5

# -------------------------------
# Helper Functions
# -------------------------------

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
    axes = [1,2,3] if K.image_data_format()=='channels_last' else [2,3,4]
    if not per_image:
        axes = [0] + axes
    return axes

def round_if_needed(x, threshold):
    if threshold is not None:
        x = tf.cast(tf.greater(x, threshold), K.floatx())
    return x

def average(x, per_image=False, class_weights=None):
    if per_image:
        x = tf.reduce_mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return tf.reduce_mean(x)

def gather_channels(gt, pr, indexes=None):
    if indexes is None:
        return gt, pr
    return _gather_channels(gt, indexes), _gather_channels(pr, indexes)

# -------------------------------
# Metric Functions
# -------------------------------

def iou_score_fn(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    intersection = tf.reduce_sum(gt * pr, axis=axes)
    union = tf.reduce_sum(gt + pr, axis=axes) - intersection
    score = (intersection + smooth) / (union + smooth)
    return average(score, per_image, class_weights)

def f_score_fn(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    fn = tf.reduce_sum(gt, axis=axes) - tp
    score = ((1 + beta**2) * tp + smooth) / ((1 + beta**2)*tp + beta**2*fn + fp + smooth)
    return average(score, per_image, class_weights)

def precision_fn(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)
    return average(score, per_image, class_weights)

def recall_fn(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    tp = tf.reduce_sum(gt * pr, axis=axes)
    fn = tf.reduce_sum(gt, axis=axes) - tp
    score = (tp + smooth) / (tp + fn + smooth)
    return average(score, per_image, class_weights)

def dice_fn(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    """Dice score per class"""
    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)
    intersection = tf.reduce_sum(gt * pr, axis=axes)
    denom = tf.reduce_sum(gt, axis=axes) + tf.reduce_sum(pr, axis=axes)
    score = (2.0 * intersection + smooth) / (denom + smooth)
    return average(score, per_image, class_weights)

def macro_dice_fn(gt, pr, class_indexes=[1,2,3], smooth=SMOOTH, threshold=None):
    """Macro Dice over specified tumor classes (default 1,2,3)"""
    dices = []
    for idx in class_indexes:
        dices.append(dice_fn(gt, pr, class_indexes=idx, smooth=smooth, threshold=threshold))
    return tf.reduce_mean(tf.stack(dices))

# -------------------------------
# Keras Metric Classes
# -------------------------------

class IOUScore(tf.keras.metrics.Metric):
    def __init__(self, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name='iou_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth
        self.iou = self.add_weight(name='iou', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = iou_score_fn(y_true, y_pred, class_weights=self.class_weights,
                             class_indexes=self.class_indexes, smooth=self.smooth,
                             per_image=self.per_image, threshold=self.threshold)
        self.iou.assign(score)

    def result(self):
        return self.iou

    def reset_state(self):
        self.iou.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights,
            "class_indexes": self.class_indexes,
            "threshold": self.threshold,
            "per_image": self.per_image,
            "smooth": self.smooth,
        })
        return config

class FScore(tf.keras.metrics.Metric):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name=None, **kwargs):
        name = name or f'f{beta}_score'
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth
        self.fscore = self.add_weight(name='fscore', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = f_score_fn(y_true, y_pred, beta=self.beta, class_weights=self.class_weights,
                           class_indexes=self.class_indexes, smooth=self.smooth,
                           per_image=self.per_image, threshold=self.threshold)
        self.fscore.assign(score)

    def result(self):
        return self.fscore

    def reset_state(self):
        self.fscore.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta,
            "class_weights": self.class_weights,
            "class_indexes": self.class_indexes,
            "threshold": self.threshold,
            "per_image": self.per_image,
            "smooth": self.smooth,
        })
        return config

class Precision(tf.keras.metrics.Metric):
    def __init__(self, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name='precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth
        self.precision = self.add_weight(name='precision', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = precision_fn(y_true, y_pred, class_weights=self.class_weights,
                             class_indexes=self.class_indexes, smooth=self.smooth,
                             per_image=self.per_image, threshold=self.threshold)
        self.precision.assign(score)

    def result(self):
        return self.precision

    def reset_state(self):
        self.precision.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights,
            "class_indexes": self.class_indexes,
            "threshold": self.threshold,
            "per_image": self.per_image,
            "smooth": self.smooth,
        })
        return config

class Recall(tf.keras.metrics.Metric):
    def __init__(self, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name='recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth
        self.recall = self.add_weight(name='recall', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = recall_fn(y_true, y_pred, class_weights=self.class_weights,
                          class_indexes=self.class_indexes, smooth=self.smooth,
                          per_image=self.per_image, threshold=self.threshold)
        self.recall.assign(score)

    def result(self):
        return self.recall

    def reset_state(self):
        self.recall.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights,
            "class_indexes": self.class_indexes,
            "threshold": self.threshold,
            "per_image": self.per_image,
            "smooth": self.smooth,
        })
        return config

class DiceScore(tf.keras.metrics.Metric):
    """Dice score per class (configurable)"""
    def __init__(self, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name='dice_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth
        self.dice = self.add_weight(name='dice', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = dice_fn(y_true, y_pred, class_weights=self.class_weights,
                        class_indexes=self.class_indexes, smooth=self.smooth,
                        per_image=self.per_image, threshold=self.threshold)
        self.dice.assign(score)

    def result(self):
        return self.dice

    def reset_state(self):
        self.dice.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights,
            "class_indexes": self.class_indexes,
            "threshold": self.threshold,
            "per_image": self.per_image,
            "smooth": self.smooth,
        })
        return config

class MacroDice(tf.keras.metrics.Metric):
    """Macro Dice averaged over multiple tumor classes"""
    def __init__(self, class_indexes=[1,2,3], threshold=None, smooth=SMOOTH, name='macro_dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.smooth = smooth
        self.macro_dice = self.add_weight(name='macro_dice', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = macro_dice_fn(y_true, y_pred, class_indexes=self.class_indexes,
                              smooth=self.smooth, threshold=self.threshold)
        self.macro_dice.assign(score)

    def result(self):
        return self.macro_dice

    def reset_state(self):
        self.macro_dice.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_indexes": self.class_indexes,
            "threshold": self.threshold,
            "smooth": self.smooth,
        })
        return config

# -------------------------------
# Aliases
# -------------------------------

iou_score = IOUScore()
f1_score = FScore(beta=1)
f2_score = FScore(beta=2)
precision = Precision()
recall = Recall()
dice_score = DiceScore()
macro_dice = MacroDice()
