import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_estimator
import mesh_tensorflow.transformer as mtf_transformer
from .optimizers import get_optimizer
from .utils import mode_to_str, get_graph_info, create_host_call, simd_mesh_setup, scalar_summary
from .dalle_mtf import DiscreteVAE, DALLE


def vae_model_fn(features, labels, mode, params):
    global_step = tf.train.get_global_step()  # Get global step

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

    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step
    features_dict = {"inputs": features, "labels": labels}
    H = W = params["dataset"]["image_size"]  # TODO: check equal
    mode_str = mode_to_str(mode)
    batch_size = params[f"{mode_str}_batch_size"]
    n_channels = params.get("input_channels", 3)
    model = DiscreteVAE(
        num_tokens=params["num_tokens"],
        batch_size=batch_size,
        dim=params["dim"],
        hidden_dim=params["hidden_dim"],
        input_channels=n_channels,
        bf_16=params.get("bf_16", True),
        num_layers=params.get("num_layers", 3),
        dimensions=H,
        params=params
    )

    mtf_features = {}
    for key, x in features_dict.items():
        if x is not None:
            x = tf.reshape(x, [batch_size, H, W, n_channels])  # NHWC
            mtf_features[key] = mtf.import_fully_replicated(
                mesh, x, mtf.Shape([
                    model.batch_dim,
                    model.height_dim,
                    model.width_dim,
                    model.channels_dim
                ]), name=key)
    scalar_summary("input_image", mtf_features["inputs"])
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

    train_gumbel = params.get("train_gumbel_hard", True)
    eval_gumbel = params.get("eval_gumbel_hard", True)

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    gumbel = train_gumbel if mode == tf.estimator.ModeKeys.TRAIN else eval_gumbel

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Gets number of microbatches per batch for serialized training
        # if param tokens_per_mb_per_replica = None, this defaults to 1 and no microbatching is performed
        num_microbatches = int(mtf_transformer.utils.serialize_num_microbatches(batch_dim=model.batch_dim,
                                                                                sequence_length=params["dataset"][
                                                                                                    "image_size"] ** 2,
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
            with tf.variable_scope("vae"):
                loss, reconstruction = model.forward(mtf_features, return_recon_loss=True, hard_gumbel=gumbel)
                return {"loss": loss, "reconstruction": reconstruction}

        # Serialize the training step - Gradients are accumulated locally and reduced once.
        var_grads, output_dict = mtf.serialize_training_step(mtf_features, serialized_fn, model.batch_dim,
                                                             num_microbatches)
        loss = output_dict["loss"]
        reconstruction = output_dict["reconstruction"]
    else:
        with tf.variable_scope("vae"):
            loss, reconstruction = model.forward(mtf_features, return_recon_loss=True, hard_gumbel=gumbel)
    scalar_summary("reconstruction_image", reconstruction)
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
    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)
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
        var_name = var.name[len(f"{scope}/"):].split(":")[0]
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
        batch_size=params[f"{mode_str}_batch_size"],
        dim=vae_params["dim"],
        hidden_dim=vae_params["hidden_dim"],
        input_channels=vae_params.get("input_channels", 3),
        bf_16=vae_params.get("bf_16", True),
        dimensions=D
    )
    return vae_model, vae_checkpoint_path


def dalle_model_fn(features, labels, mode, params):
    # since we can simply infer labels here based on the input - features here are the text input,
    # and labels are the image input
    global_step = tf.train.get_global_step()  # Get global step

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

    H = W = params["dataset"]["image_size"]
    mode_str = mode_to_str(mode)
    Vae, vae_checkpoint_path = load_vae_model(params, mode_str)
    image_seq_len = (Vae.H // (2 ** Vae.num_layers)) ** 2  # TODO: check this is correct
    batch_size = params[f"{mode_str}_batch_size"]
    n_channels = params.get("input_channels", 3)
    model = DALLE(
        n_embd=params["n_embd"],
        text_vocab_size=params["text_vocab_size"],
        image_vocab_size=params["image_vocab_size"],
        text_seq_len=params["text_seq_len"],
        image_seq_len=image_seq_len,
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        batch_size=batch_size,
        bf_16=params["bf_16"],
        mode=mode_str,
        params=params,
        vae=Vae
    )

    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step
    features_dict = {"image_inputs": features,
                     "text_inputs": labels}
    mtf_features = {}
    for key, x in features_dict.items():
        if x is not None:
            if key == "text_inputs":
                x = tf.reshape(x, [batch_size, params["text_seq_len"]])
                mtf_shape = mtf.Shape([model.dimensions["batch_dim"], model.dimensions["text_sequence_dim"]])
            elif key == "image_inputs":
                x = x[0] # TODO: why am i getting a tuple of inputs here? something's not right
                mtf_shape = mtf.Shape([
                    model.dimensions["batch_dim"],
                    Vae.height_dim,
                    Vae.width_dim,
                    Vae.channels_dim
                ])
                x = tf.reshape(x, [batch_size, H, W, n_channels])  # NHWC

            mtf_features[key] = mtf.import_fully_replicated(
                mesh, x, mtf_shape, name=key)
    scalar_summary("input_image", mtf_features["image_inputs"])
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Gets number of microbatches per batch for serialized training
        # if param tokens_per_mb_per_replica = None, this defaults to 1 and no microbatching is performed
        num_microbatches = int(mtf_transformer.utils.serialize_num_microbatches(batch_dim=model.dimensions["batch_dim"],
                                                                                sequence_length=model.total_seq_len,
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
            loss, loss_batch = model.forward(mtf_features, return_loss=True)
            return {"loss": loss, "loss_batch": loss_batch}

        # Serialize the training step - Gradients are accumulated locally and reduced once.
        var_grads, output_dict = mtf.serialize_training_step(mtf_features, serialized_fn, model.dimensions["batch_dim"],
                                                             num_microbatches)
        loss = output_dict["loss"]
        loss_batch = output_dict["loss_batch"]
    else:
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
    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=False)

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
        
    # now we can load the vae weights
    initialize_vae_weights(vae_checkpoint_path)

    with mtf.utils.outside_all_rewrites():
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
