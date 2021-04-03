import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_estimator
import mesh_tensorflow.transformer as mtf_transformer
from .optimizers import get_optimizer
from .utils import mode_to_str, get_graph_info, create_host_call, simd_mesh_setup, scalar_summary
from .dalle_mtf import DALLE, sample_autoregressive
from .vae_tf import DiscreteVAE
from tensorflow.python.ops import resources


def initialize_vae_weights(checkpoint_path, scope="vae"):
    """
    Initialize the vae model from the checkpoint.
    This function will be called after the graph has been constructed.
    """
    vars_to_restore = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    ckpt_vars = [
        name for name, _ in tf.train.list_variables(checkpoint_path)]
    tf.logging.info(f"RESTORING {len(vars_to_restore)} VAE VARS FROM CHECKPOINT: ")
    tf.logging.info(f"CHECKPOINT PATH: {checkpoint_path}")
    tf.logging.info(f"CHECKPOINT VARS:")
    tf.logging.info(ckpt_vars)
    vae_load_dict = {}
    for var in vars_to_restore:
        var_name = var.name.split(":")[0]
        tf.logging.info(var_name)
        tf.logging.info(var)
        vae_load_dict[var_name] = var

    # Initialize vae weights
    tf.train.init_from_checkpoint(checkpoint_path, vae_load_dict)


def load_vae_model(params, mode_str):
    vae_checkpoint_path = params.get("vae_checkpoint_path")
    vae_params = params.get("vae_params")
    assert vae_params is not None, "vae model config must be supplied"
    if vae_checkpoint_path is None:
        vae_checkpoint_path = tf.train.latest_checkpoint(vae_params["model_path"])
    assert vae_checkpoint_path is not None, "pretrained vae needed for training"
    D = params["dataset"]["image_size"]
    vae_model = DiscreteVAE(
        num_tokens=vae_params["num_tokens"],
        dim=vae_params["dim"],
        hidden_dim=vae_params["hidden_dim"],
        input_channels=vae_params.get("input_channels", 3),
        convblocks=params.get("vae_params").get("convblocks", [(3, 64), (3, 128), (3, 256)]),
        stack_factor=params.get("vae_params").get("stack_factor", 1),
        dimensions=D
    )
    return vae_model, vae_checkpoint_path


def dalle_model_fn(features, labels, mode, params):
    # since we can simply infer labels here based on the input - features here are the text input,
    # and labels are the image input
    global_step = tf.train.get_global_step()  # Get global step

    mode_str = mode_to_str(mode)

    # load vae in tensorflow graph before mtf
    vae, vae_checkpoint_path = load_vae_model(params, mode_str)

    H = W = params["dataset"]["image_size"]
    batch_size = params[f"{mode_str}_batch_size"]
    n_channels = params.get("input_channels", 3)
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:

        with tf.variable_scope("vae"):
            vae_logits = vae.forward(features, return_logits=True)

        # TODO: using argmax sampling for now, but is that optimal?
        tokens = tf.math.argmax(vae_logits, -1)
        img_tokens_reshaped = tf.cast(tf.reshape(tokens, (batch_size, params['image_seq_len'])), tf.int32)

        # TODO: get rid of this ugly hack, its just to pull the decoder parameters in during training
        with tf.variable_scope('vae'):
            vae.decoder(tf.zeros_like(vae_logits))

    # Construct mtf graph + mesh from params
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    layout_rules = mtf.convert_to_layout_rules(params["layout"])

    # Mesh setup
    if params["use_tpu"]:
        var_placer, mesh_impl = simd_mesh_setup(params, mesh_shape, layout_rules)
    else:
        var_placer = None
        gpu_ids = params["gpu_ids"]
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            mesh_shape, layout_rules, gpu_ids)

    # Build mtf mesh object
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    model = DALLE(
        n_embd=params["n_embd"],
        text_vocab_size=params["text_vocab_size"],
        image_vocab_size=params["image_vocab_size"],
        text_seq_len=params["text_seq_len"],
        image_seq_len=params['image_seq_len'],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        batch_size=batch_size,
        bf_16=params["bf_16"],
        mode=mode_str,
        params=params,
    )

    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        features_dict = {"image_inputs": features,
                         "text_inputs": labels}
        mtf_features = {}
        for key, x in features_dict.items():
            if x is not None:
                if key == "text_inputs":
                    text_tokens = tf.reshape(x, [batch_size, params["text_seq_len"]])
                    x = tf.concat((text_tokens, model.shift_image_tokens(img_tokens_reshaped)), axis=1)
                    mtf_shape = mtf.Shape([model.dimensions["batch_dim"], model.dimensions["total_seq_dim"]])

                    mtf_features["tokens"] = mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)

                if key == "image_inputs":
                    mtf_shape = mtf.Shape([
                        model.dimensions["batch_dim"],
                        mtf.Dimension("img_height_dim", vae.H),
                        mtf.Dimension("img_width_dim", vae.W),
                        mtf.Dimension("img_channel_dim", vae.num_ch),
                    ])
                    x = tf.reshape(x, [batch_size, H, W, n_channels])  # NHWC
                    mtf_features["image_inputs"] = mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)
        denormalize = lambda x: (x + 1) / 2
        scalar_summary("input_image", denormalize(mtf_features["image_inputs"]))
    else:
        features_dict = {"text_inputs": labels}
        mtf_features = {}
        for key, x in features_dict.items():
            if x is not None:
                if key == "text_inputs":
                    text_tokens = tf.reshape(x, [batch_size, params["total_seq_len"]])
                    mtf_shape = mtf.Shape([model.dimensions["batch_dim"], model.dimensions["total_seq_dim"]])
                    mtf_features["tokens"] = mtf.import_fully_replicated(mesh, text_tokens, mtf_shape, name=key)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Set up the model for prediction
        inputs = mtf_features["tokens"]

        mtf_samples = sample_autoregressive(inputs,
                                            model,
                                            max_steps=model.total_seq_dim, # will always run until the full image is produced
                                            stop_at_token=None,
                                            temperature=0.9,
                                            padding_id = 0,
                                            variable_dtype=model.variable_dtype,
                                            has_partial_sequences=True,
                                            remove_partial_sequences=True,
                                            sampling_keep_top_k=-1,
                                            )

        mtf_samples = mtf.anonymize(mtf_samples)
        inputs = mtf.anonymize(inputs)
        lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=params.get('autostack', True))

        inputs = lowering.export_to_tf_tensor(inputs)
        outputs = lowering.export_to_tf_tensor(mtf_samples)

        initialize_vae_weights(vae_checkpoint_path)

        img_outputs = model.unshift_image_tokens(outputs[:, -model.image_seq_len:])

        with tf.variable_scope('vae'):
            predictions_decoded = vae.decode(img_outputs)

        predictions = {
            "inputs": inputs,
            "outputs": img_outputs,
            "predictions_decoded": predictions_decoded
        }

        def scaffold_fn():
            return tf.train.Scaffold(
                local_init_op=tf.group(
                    tf.train.Scaffold.default_local_init_op(),
                    lowering.copy_masters_to_slices(),
                    name="mtf_local_init_op"),
                ready_op=tf.concat(
                    [tf.report_uninitialized_variables(),
                     resources.report_uninitialized_resources()],
                    axis=0,
                    name="mtf_ready_op"))

        return tpu_estimator.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            scaffold_fn=scaffold_fn,
            prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Gets number of microbatches per batch for serialized training
        # if param tokens_per_mb_per_replica = None, this defaults to 1 and no microbatching is performed
        num_microbatches = int(mtf_transformer.utils.serialize_num_microbatches(batch_dim=model.dimensions["batch_dim"],
                                                                                sequence_length=model.total_seq_dim,
                                                                                mesh_shape=mesh_shape,
                                                                                layout_rules=layout_rules,
                                                                                tokens_per_microbatch_per_replica=
                                                                                params[
                                                                                    "tokens_per_mb_per_replica"]))
    else:
        num_microbatches = 1

    params["num_microbatches"] = num_microbatches  # Add num microbatches to params

    if num_microbatches > 1:
        # For serialize_training_step we need to modify the model to output results in a dict
        def serialized_fn(mtf_features):
            with tf.variable_scope('dall-e'):
                loss, loss_batch = model.forward(mtf_features, return_loss=True)
                return {"loss": loss, "loss_batch": loss_batch}

        # Serialize the training step - Gradients are accumulated locally and reduced once.
        var_grads, output_dict = mtf.serialize_training_step(mtf_features, serialized_fn, model.dimensions["batch_dim"],
                                                             num_microbatches)
        loss = output_dict["loss"]
        loss_batch = output_dict["loss_batch"]
    else:
        with tf.variable_scope('dall-e'):
            loss, loss_batch = model.forward(mtf_features, return_loss=True)

    del loss_batch  # TODO: may need this for some metrics - otherwise, remove from output

    if mode == tf.estimator.ModeKeys.TRAIN:
        # In TRAIN mode, get optimizer
        if num_microbatches > 1:
            # If we are splitting the batch into microbatches, var grads are created in the serialize_training_step fn
            # So we pass them in here
            _, update_ops, var_grads = get_optimizer(mesh, loss, params, variable_dtype=model.variable_dtype,
                                                     inp_var_grads=var_grads)
        else:
            # Otherwise, they are created in the get_optimizer fn, so we leave inp_var_grads blank
            _, update_ops, var_grads = get_optimizer(mesh, loss, params, variable_dtype=model.variable_dtype)
        # Log summaries to tensorboard
        scalar_summary("loss", loss)

    # Gets & prints info about no. trainable vars in the model & dimension names
    get_graph_info(graph)

    # 'lowers' mtf tensors into a tf graph - this enables us to export results as tf tensors
    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=params.get('autostack', True))

    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.cast(tf_loss, tf.float32)


    if mode == tf.estimator.ModeKeys.TRAIN:
        # Use our patched version until mtf updates theirs
        host_call = create_host_call(params['model_path'])
        mtf.utils.remove_summaries()

        # Creates train_op
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))  # Need to manually increment global_step
        train_op = tf.group(tf_update_ops)
        

    with mtf.utils.outside_all_rewrites():
        # only *now* can we initialize vae weights (stupid tensorflow)
        initialize_vae_weights(vae_checkpoint_path)

        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Set up the checkpoint server and return the TPUEstimatorSpec
            saver = tf.train.Saver(
                tf.global_variables(),
                sharded=True,
                max_to_keep=params.get("max_checkpoints", 5),
                keep_checkpoint_every_n_hours=2,
                defer_build=False,
                save_relative_paths=True)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            saver_listener = mtf.MtfCheckpointSaverListener(lowering)
            saver_hook = tf.train.CheckpointSaverHook(
                params["model_path"],
                save_steps=params["steps_per_checkpoint"],
                saver=saver,
                listeners=[saver_listener])

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.TRAIN,
                loss=tf_loss,
                host_call=host_call,
                train_op=train_op,
                training_hooks=[restore_hook, saver_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.EVAL,
                evaluation_hooks=[restore_hook],
                loss=tf_loss,
                eval_metrics=None)
