import tensorflow.compat.v1 as tf


def gumbel_softmax(logits, axis, temperature=1, hard=True):
    with tf.name_scope(name="gumbel_softmax"):
        smol_val = 1e-9
        logits = tf.cast(logits, tf.float32)
        g = -tf.log(-tf.log(
            tf.random_uniform(
                logits.shape,
                minval=smol_val,
                maxval=1.,
                dtype=logits.dtype)))
        logits += g
        sample = tf.nn.softmax(logits/temperature, axis, name="gumbel_softmax_softmax")

        if hard:
            sample_hard = tf.cast(tf.one_hot(tf.argmax(sample, axis), sample.shape[axis], axis=axis), sample.dtype)
            sample = tf.stop_gradient(sample_hard - sample) + sample

        return sample


def mse_loss(prediction, target):
    return tf.reduce_mean(tf.square(prediction - target))
