import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


def gumbel_softmax(logits, dim, temperature=1, hard=True):
    with tf.name_scope(name="gumbel_softmax"):
        smol_val = 1e-9
        logits = mtf.cast(logits, tf.float32)
        g = -mtf.log(-mtf.log(
            mtf.random_uniform(
                logits.mesh,
                logits.shape,
                minval=smol_val,
                maxval=1.,
                dtype=logits.dtype)))
        logits += g * temperature
        sample = mtf.softmax(logits, dim, name="gumbel_softmax_softmax")

        if hard:
            sample_hard = mtf.cast(mtf.one_hot(mtf.argmax(sample, dim), dim), sample.dtype)
            sample = mtf.stop_gradient(sample_hard - sample) + sample

        return sample


def mse_loss(prediction, target):
    return mtf.reduce_mean(mtf.square(prediction - target))


def norm(x, axis, epsilon=1e-8):
    x -= mtf.reduce_mean(x, reduced_dim=axis, name="norm_reduce_mean_u")
    s = mtf.reduce_mean(mtf.square(x), reduced_dim=axis, name="norm_reduce_mean_s")
    return x * mtf.rsqrt(s + epsilon)
