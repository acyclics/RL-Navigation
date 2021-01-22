import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
import tensorflow as tf

from architecture.distributions.diaggauss import DiagGaussian


class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class Navigation(tf.keras.Model):

    def __init__(self, batch_size=1, training=True):
        super(Navigation, self).__init__()
        self.batch_size = batch_size
        self.training = training
        self.diagguass = DiagGaussian()

        self.core = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
            tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
        ])

        with tf.name_scope("xyyaw"):
            self.act_core = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            ])
            self.xyyaw_mean = tf.keras.layers.Dense(6, kernel_initializer=normc_initializer(0.01), activation=tf.nn.tanh, name="xyyaw")
            self.xyyaw_logstd = tf.Variable(name="logstd_xyyaw", initial_value=tf.zeros_initializer()([6]), trainable=True)
            #self.xyyaw_logstd = tf.keras.layers.Dense(6, kernel_initializer=normc_initializer(0.01), name="xyyaw")

        with tf.name_scope("value"):
            self.val_core = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01)),
                tf.keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            ])
            self.value = tf.keras.layers.Dense(1, name="value", activation=None, kernel_initializer=normc_initializer(1.0))

    @tf.function
    def call(self, obs):
        core_output = self.core(obs)

        with tf.name_scope("xyyaw"):
            act_core = self.act_core(core_output)
            xyyaw_mean = self.xyyaw_mean(act_core)
            actions = self.diagguass.sample(xyyaw_mean, self.xyyaw_logstd)
        
        with tf.name_scope('value'):
            val_core = self.val_core(core_output)
            value = self.value(val_core)[:, 0]   # flatten value otherwise it might broadcast
                
        neglogp = self.diagguass.neglogp(xyyaw_mean, self.xyyaw_logstd, actions)
        entropy = self.diagguass.entropy(self.xyyaw_logstd)
         
        return actions, neglogp, entropy, value, xyyaw_mean, self.xyyaw_logstd
    
    def call_build(self):
        """
        IMPORTANT: This function has to be editted so that the below input features
        have the same shape as the actual inputs, otherwise the weights would not
        be restored properly.
        """
        self(np.zeros([self.batch_size, 22]))
 