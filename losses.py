import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import numpy as np

"""
Softmax losses
"""
def EarthMoversDistance():
    def earth_movers_distance(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        cdf_true = K.cumsum(y_true, axis=-1)
        cdf_pred = K.cumsum(y_pred, axis=-1)
        return K.mean(K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1)))
    
    return earth_movers_distance


class Statistics():
    def __init__(self, n):
        self.scores = tf.constant(np.arange(1,n+1), shape=[n,1], dtype=tf.float32)

    def loss(self, loss, mode, name):
        stat = getattr(self, mode)
        name = name + ("" if mode == "mean" else "_" + mode)

        def wrapped_loss(y_true, y_pred):
            y_true = stat(tf.convert_to_tensor(y_true, tf.float32))
            y_pred = stat(tf.convert_to_tensor(y_pred, tf.float32))
            return loss(y_true, y_pred)

        exec(f"def {name}(y_true, y_pred): return wrapped_loss(y_true, y_pred)",
            {"wrapped_loss" : wrapped_loss}, locals()
        )
        return locals()[name]

    def moment(self, x, m):
        return tf.linalg.matmul(x, self.scores**m)
    
    def mean(self, x):
        return self.moment(x, 1)
    
    def var(self, x):
        return tf.abs(self.moment(x, 2)-(self.mean(x)**2))
    
    def std(self, x):
        return tf.sqrt(self.var(x))
    
    #Fischer skewness
    def skew(self, x):
        return (self.moment(x, 3)-(3*self.mean(x)*(self.std(x)**2))-(self.mean(x)**3))/(self.std(x)**3)

def MeanAbsoluteError(n, mode="mean"):
    s = Statistics(n)
    return s.loss(
        lambda y_true, y_pred : tf.reduce_mean(tf.abs(y_true - y_pred)),
        mode,
        "mean_absolute_error"
    )

def RootMeanSquaredError(n, mode="mean"):
    s = Statistics(n)
    return s.loss(
        lambda y_true, y_pred : tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))),
        mode,
        "root_mean_squared_error"
    )

def MeanSquaredError(n, mode="mean"):
    s = Statistics(n)
    return s.loss(
        lambda y_true, y_pred : tf.reduce_mean(tf.square(y_true - y_pred)),
        mode,
        "mean_squared_error"
    )

def PearsonCorrelation(n, mode="mean"):
    s = Statistics(n)
    return s.loss(
        tfp.stats.correlation,
        mode,
        "correlation"
    )