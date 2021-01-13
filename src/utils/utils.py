import json
from collections import defaultdict
from urllib.parse import urlparse
from shutil import rmtree
import os
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import mesh_tensorflow as mtf
import logging 
import sys
from mesh_tensorflow.ops import Operation, Tensor

def fetch_model_params(model):
    model_path = model if model.endswith(".json") else f"./configs/{model}.json"
    with open(model_path) as f:
        params = json.load(f)
    return defaultdict(lambda: None, params)


def yes_or_no(question):
    while True:
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def mode_to_str(mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return "predict"
    elif mode == tf.estimator.ModeKeys.EVAL:
        return "eval"
    elif mode == tf.estimator.ModeKeys.TRAIN:
        return "train"
    else:
        raise ValueError(f"Invalid mode {mode}")


def remove_gs_or_filepath(path):
    parsed_url = urlparse(path)
    if parsed_url.scheme == "gs":
        os.system(f"gsutil rm -rf {path}")
        return
    rmtree(path)


def maybe_remove_gs_or_filepath(path):
    if yes_or_no(f"Are you sure you want to remove '{path}' to start afresh?"):
        remove_gs_or_filepath(path)
    else:
        exit()


def get_n_trainable_vars(graph):
    """
    Gets number of trainable vars in a MTF model.

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    total_parameters = 0
    for variable in graph.trainable_variables:
        shape = variable.shape.dims
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.size
        total_parameters += variable_parameters
    print(f"\n\nN PARAMS:\n{total_parameters:,}\n\n")


def print_dim_names(graph):
    """
    Print names of all Dimensions
    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    all_dim_names = []
    for variable in graph.all_variables:
        names = variable.shape.dimension_names
        all_dim_names.append(names)

    # Print all dim names in graph & write to file
    all_dim_names = [item for sublist in all_dim_names for item in sublist]  # Flatten all dims
    unique_dims = list(set(all_dim_names))
    print("ALL DIM NAMES:")
    for dim_name in unique_dims:
        print(dim_name)
    print('\n')


def get_graph_info(graph):
    """
    Wrapper fn that calculates number of trainable vars in an MTF graph & prints all dim_names to file

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    get_n_trainable_vars(graph)
    print_dim_names(graph)


def create_host_call(model_dir):
    """Construct a host_call writing scalar summaries.

    Borrowed from t2t.

    Args:
        model_dir: String containing path to train
    Returns:
        (fn, args) Pair to be called by TPUEstimator as the host_call.
    """

    graph = tf.get_default_graph()
    # A list of (name, lowered tensor) tuples
    summaries = graph.get_collection(mtf.utils.SCALAR_SUMMARIES_COLLECTION_KEY)

    def maybe_cast(tensor):
        # assert tensor.shape.is_compatible_with([]), tensor.name
        if tensor.dtype == tf.int64:
            return tf.to_int32(tensor)
        if tensor.dtype == tf.bfloat16:
            return tf.cast(tensor, tf.float32)
        return tensor

    reshaped_tensors = []
    for _, t in summaries:
        try:
            t = tf.reshape(maybe_cast(t), [1])
        except:
            pass
        reshaped_tensors.append(t)
    # When no supported summaries are found, don't create host_call. Otherwise,
    # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
    # it, eventually causing hang.
    if not reshaped_tensors:
        return None

    def host_call_fn(global_step, *args):
        """Training host call. Creates scalar summaries for training metrics."""
        # This function is executed on the CPU and should not directly reference
        # any Tensors in the rest of the `model_fn`. To pass Tensors from the
        # model to the `model_fn`, provide as part of the `host_call`.
        global_step = tf.cast(global_step[0], tf.int64)
        with tf2.summary.create_file_writer(model_dir).as_default():
            # We cannot directly use any tensor from summaries, because each
            # tensor here must be a concat of multiple tensors from all shards.
            # Therefore, we rely on the assumption that args wil have the same
            # length as summaries, and all tensors in args will have the same
            # order of self._tup_summaries.
            assert len(args) == len(summaries)
            for i, tensor in enumerate(args):
                name = summaries[i][0]
                if not "image" in name:
                    tf2.summary.scalar(name, tf.reduce_mean(tensor), step=global_step)
                else:
                    tf2.summary.image(name, tensor, step=global_step)
        return tf.summary.all_v2_summary_ops()

    global_step_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
    return host_call_fn, [global_step_t] + reshaped_tensors

def simd_mesh_setup(params, mesh_shape, layout_rules):
    """Constructs SimdMesh function - instructions on how to evenly split tensors across all TPU cores"""

    num_hosts = params["context"].num_hosts
    host_placement_fn = params["context"].tpu_host_placement_function
    device_list = [host_placement_fn(host_id=i) for i in range(num_hosts)]
    tf.logging.info(f"device_list = {device_list}")

    # TODO: Better estimation of replica cache size?
    replica_cache_size = 300 * 1000000  # 300M per replica

    # Worker 0 caches all the TPU binaries
    worker0_mem = replica_cache_size * params["context"].num_replicas
    devices_memory_usage = [worker0_mem] + [0] * (num_hosts - 1)
    var_placer = mtf.utils.BalancedVariablePlacer(device_list, devices_memory_usage)
    mesh_devices = [""] * mesh_shape.size
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        mesh_shape, layout_rules, mesh_devices, params["context"].device_assignment)

    return var_placer, mesh_impl

def setup_logging(args, logdir="logs"):
    os.makedirs(logdir, exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    tf.get_logger().propagate = False  # Remove double log on console
    name = os.path.splitext(os.path.basename(args.model))[0]
    handlers = [
        logging.FileHandler(f"logs/{name}.log"),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger("tensorflow")
    logger.handlers = handlers
    return logger

class ScalarSummaryOperation(Operation):
  """Similar to tf.Print."""

  def __init__(self, name, x):
    super(ScalarSummaryOperation, self).__init__(
        [x], x.mesh, name=name)
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def lower(self, lowering):
    lowered_input = lowering.tensors[self.inputs[0]].to_laid_out_tensor()
    tf.add_to_collection(mtf.utils.SCALAR_SUMMARIES_COLLECTION_KEY,
                         (self.name, lowered_input.tensor_list[0]))
    lowering.set_tensor_lowering(
        self.outputs[0], lowered_input)

  def gradient(self, grad_ys):
    return grad_ys


def scalar_summary(name, x):
  """Call tf.summary.scalar.
  Caveat - summaries do not generally work on TPU - they need to be rewritten
  into a host call.
  TODO(noam): provide a pointer to code for this.
  Args:
    name: a string
    x: a 0-dimensional Tensor
  Returns:
    a Tensor which is identical in value to x
  """
  return ScalarSummaryOperation(name, x)

def get_image_seq_len(dalle_params):
    return (dalle_params["vae_params"]['dataset']['image_size'] // (2 ** len(dalle_params["vae_params"]['convblocks']))) ** 2 // (
                dalle_params.get("vae_params").get("stack_factor", 1) ** 2)