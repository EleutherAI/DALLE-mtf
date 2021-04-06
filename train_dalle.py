from functools import partial
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow_estimator.python.estimator import estimator as estimator_lib
import argparse
from src.utils import *
from src.model_fns import dalle_model_fn
from src.input_fns import dalle_input_fn, pred_input, pred_output
from src.data import get_tokenizer

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, if any.")
    parser.add_argument("--gpu_ids", nargs="+", type=str, default=["device:GPU:0"],
                        help="If training on GPU, can specify your GPU names in a list - i.e 'device:GPU:0 device:GPU:1'")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")
    parser.add_argument('--predict', action='store_true', help='run model in predict mode')
    parser.add_argument('--prompt', type=str, default='face')
    args = parser.parse_args()
    assert args.model is not None, "Model must be set"
    return args


def main():
    # parse args and params
    args = parse_args()
    logging = setup_logging(args)
    params = fetch_model_params(args.model)
    params["vae_params"] = fetch_model_params(params["vae_model"])
    save_config(params, params['model_dir'])
    assert params["model_type"].lower() == "dalle", f'model_type {params["model_type"]} not recognized'

    # Confirm deletion of checkpoint files if --new flag is set
    if args.new:
        maybe_remove_gs_or_filepath(params["model_path"])

    # get current step
    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params["model_path"]))
    logging.info(f"Current step: {current_step}")

    # Add to params:
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    params["num_cores"] = mesh_shape.size
    params["use_tpu"] = True if not args.tpu is None else False
    params["gpu_ids"] = args.gpu_ids
    tokenizer = get_tokenizer(params["tokenizer"])
    assert len(tokenizer) == params["text_vocab_size"], f"tokenizer vocab size {len(tokenizer)} must equal model vocab size {params['text_vocab_size']}"
    params['image_seq_len'] = get_image_seq_len(params)
    params['total_seq_len'] = params['image_seq_len'] + params['text_seq_len']
    params["padding_id"] = tokenizer.encode(tokenizer.pad_token)[0]
    # Set up TPUs and Estimator
    if args.tpu == "colab":
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() if params["use_tpu"] else None
    else:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu) if params[
            "use_tpu"] else None

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params["model_path"],
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=params["iterations"],
        save_summary_steps=params["iterations"],
        tpu_config=tpu_config.TPUConfig(
            num_shards=mesh_shape.size,
            iterations_per_loop=params["iterations"],
            num_cores_per_replica=1,
            experimental_host_call_every_n_steps=100,
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

    estimator = tpu_estimator.TPUEstimator(
        use_tpu=params["use_tpu"],
        model_fn=dalle_model_fn,
        config=config,
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["eval_batch_size"],
        predict_batch_size=params["predict_batch_size"],
        params=params)
    if args.predict:
        # Predict
        pred_input_fn = partial(pred_input, tokenizer=tokenizer, prompt=args.prompt)
        predictions = estimator.predict(input_fn=pred_input_fn)
        logging.info("Predictions generated")
        pred_output(predictions, 'test')
        return

    has_predict_or_eval_steps = params["predict_steps"] > 0 or params["eval_steps"] > 0
    if has_predict_or_eval_steps:
        # Eval and train - stop and predict and/or eval every checkpoint
        while current_step < params["train_steps"]:
            next_checkpoint = min(current_step + params["steps_per_checkpoint"], params["train_steps"])
            estimator.train(input_fn=partial(dalle_input_fn, eval=False),
                            max_steps=next_checkpoint)
            current_step = next_checkpoint
            if params["predict_steps"] > 0:
                raise NotImplementedError
            if params["eval_steps"] > 0:
                estimator.evaluate(input_fn=partial(dalle_input_fn, eval=True),
                            steps=params["eval_steps"])
        return
    else:
        # Else, just train
        while current_step < params["train_steps"]:
            # Else, don't stop and restart
            estimator.train(input_fn=partial(dalle_input_fn, eval=False),
                            max_steps=params["train_steps"])



if __name__ == "__main__":
    tf.disable_v2_behavior()
    main()
