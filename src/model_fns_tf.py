import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.tpu import tpu_estimator
from .vae_tf import DiscreteVAE
from .utils import mode_to_str


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
        recompute_grad=params.get("recompute_grad", False),
        use_bf16=params.get("use_bf16", False),
        stack_factor=params.get("stack_factor", 1),
        dimensions=H
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

    train_gumbel = params.get("train_gumbel_hard", True)
    eval_gumbel = params.get("eval_gumbel_hard", True)

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    gumbel = train_gumbel if mode == tf.estimator.ModeKeys.TRAIN else eval_gumbel

    if params.get("temp_anneal_steps", None):
        warmup_frac = tf.cast(tf.train.get_global_step(), tf.float32) / params["temp_anneal_steps"]
        warmup_frac = tf.minimum(warmup_frac, tf.constant(1.0))
        temp = params["temp_start"] - warmup_frac * (params["temp_start"] - params["temp"])
    else:
        temp = params.get("temp", 1.0)

    # TODO: add back in microbatching
    if params.get("use_bf16", False):
        with tf.tpu.bfloat16_scope():
            with tf.variable_scope("vae"):
                loss, reconstruction = model.forward(features, return_recon_loss=True, temperature=temp, hard_gumbel=gumbel)
                loss = tf.cast(loss, tf.float32)
                reconstruction = tf.cast(reconstruction, tf.float32)
    else:
        with tf.variable_scope("vae"):
            loss, reconstruction = model.forward(features, return_recon_loss=True, temperature=temp, hard_gumbel=gumbel)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=params["lr"]
    )
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    def host_call_fn(gs, loss, input, reconstruction):
        gs = gs[0]
        loss = tf.math.reduce_mean(loss)
        denormalize = lambda x: (x + 1) / 2

        with tf2.summary.create_file_writer(params['model_path']).as_default():
            tf2.summary.scalar('loss', loss, step=gs)
            tf2.summary.image('input_image', denormalize(input), step=gs)
            tf2.summary.image('reconstruction_image', denormalize(reconstruction), step=gs)

            return tf.summary.all_v2_summary_ops()

    def metric_fn(gs, loss, input, reconstruction):
        gs = gs[0]
        loss = tf.math.reduce_mean(loss)
        denormalize = lambda x: (x + 1) / 2

        with tf2.summary.create_file_writer(params['model_path']).as_default():
            loss_op = tf.metrics.mean(loss)

            with tf2.summary.record_if(loss_op[0] < tf.constant(1e-9)):
                tf2.summary.image('eval/input_image', denormalize(input), step=gs)
                tf2.summary.image('eval/reconstruction_image', denormalize(reconstruction), step=gs)

            with tf.control_dependencies(tf.summary.all_v2_summary_ops()):
                dummy_op = tf.no_op()

            return {"_loss": loss_op,
                    "zzz_dummy": (tf.constant(0), dummy_op)}

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    gs_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(loss, [1])

    host_call = (host_call_fn, [gs_t, loss_t, features, reconstruction])
    metric = (metric_fn, [gs_t, loss_t, features, reconstruction])

    return tpu_estimator.TPUEstimatorSpec(
        mode,
        loss=loss,
        host_call=host_call if mode == tf.estimator.ModeKeys.TRAIN else None,
        train_op=train_op,
        eval_metrics=metric)
