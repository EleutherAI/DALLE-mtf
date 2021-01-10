import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.tpu import tpu_estimator
from .optimizers import get_optimizer
from .vae_tf import DiscreteVAE
from .utils import scalar_summary, mode_to_str, create_host_call


def vae_model_fn(features, labels, mode, params):
    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step

    H = W = params["dataset"]["image_size"]  # TODO: check equal
    mode_str = mode_to_str(mode)
    batch_size = params[f"{mode_str}_batch_size"]
    n_channels = params.get("input_channels", 3)
    model = DiscreteVAE(
        num_tokens=params["num_tokens"],
        dim=params["n_embd"],
        hidden_dim=params["hidden_dim"],
        input_channels=n_channels,
        convblocks=params.get("convblocks", [(3, 64), (3, 128), (3, 256)]),
        dimensions=H
    )

    # mtf_features = {}
    # for key, x in features_dict.items():
    #     if x is not None:
    #         x = tf.reshape(x, [batch_size, H, W, n_channels])  # NHWC
    #         mtf_features[key] = x
    # scalar_summary("input_image", features_dict["inputs"])
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

    train_gumbel = params.get("train_gumbel_hard", True)
    eval_gumbel = params.get("eval_gumbel_hard", True)

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    gumbel = train_gumbel if mode == tf.estimator.ModeKeys.TRAIN else eval_gumbel

    # TODO: add back in microbatching
    with tf.variable_scope("vae"):
        loss, reconstruction = model.forward(features, return_recon_loss=True, hard_gumbel=gumbel)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=params["lr"]
    )
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    # scalar_summary("reconstruction_image", reconstruction)
    # Log summaries to tensorboard
    # scalar_summary("loss", loss)

    def host_call_fn(gs, loss, input, reconstruction):
        """Training host call. Creates scalar summaries for training metrics.
        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.
        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.
        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.
        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        loss = tf.math.reduce_mean(loss)

        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf2.summary.create_file_writer(params['model_path']).as_default():
            tf2.summary.scalar('loss', loss, step=gs)
            tf2.summary.image('input_image', input, step=gs)
            tf2.summary.image('reconstruction_image', reconstruction, step=gs)

            return tf.summary.all_v2_summary_ops()

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    gs_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(loss, [1])

    host_call = (host_call_fn, [gs_t, loss_t, features, reconstruction])

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call)
