from keras.metrics import Metric
import tensorflow as tf
import keras

class MCC(Metric):
    def __init__(self, name='mcc', **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        numerator = (tp * tn) - (fp * fn)
        denominator = tf.math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        return tf.where(tf.equal(denominator, 0), 0.0, numerator / denominator)

    def reset_states(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)



def custom_binary_cross_entropy(y_true, y_pred):
    epsilon = keras.epsilon()  # petite valeur pour éviter log(0)
    y_pred = keras.clip(y_pred, epsilon, 1.0 - epsilon)
    return -keras.mean(y_true * keras.log(y_pred) + (1 - y_true) * keras.log(1 - y_pred))