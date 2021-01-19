import mesh_tensorflow as mtf
from mesh_tensorflow.ops import Operation, Dimension, Shape, Tensor, mtf_slice
import tensorflow.compat.v1 as tf


class CustomPadOperation(Operation):
    """tf.pad.
  Similar to tf.pad but we only pad along one axis given by pad_dim_name
  with values specified by paddings. paddings is a list of two
  values, giving the padding value before and after pad_dim.
  """

    def __init__(self, x, paddings, pad_dim_name, pad_value=0, name=None):
        super(CustomPadOperation, self).__init__([x], name=name or "pad")
        assert len(paddings) == 2
        input_shape = self._inputs[0].shape
        dim_names = [dim.name for dim in x.shape.dims]
        if pad_dim_name not in dim_names:
            raise ValueError("Padding dim name %s not found in input." % pad_dim_name)
        self._paddings = paddings
        self._axis = axis = dim_names.index(pad_dim_name)
        output_size = input_shape.dims[axis].size + sum(paddings)
        self._output_dim = Dimension(pad_dim_name, output_size)
        output_shape = Shape(
            input_shape.dims[:axis] +
            [self._output_dim] + input_shape.dims[axis + 1:])
        self._outputs = [Tensor(self, output_shape, x.dtype)]
        self._splittable_dims, self._unsplittable_dims = (
            self._initialize_splittable_and_unsplittable_dims(
                "splittable", [pad_dim_name]))
        self.pad_value = pad_value

    def gradient(self, grad_ys):
        slice_dim_name = self._output_dim.name
        slice_size = self._inputs[0].shape.dims[self._axis].size
        return [mtf_slice(grad_ys[0], self._paddings[0],
                          slice_size, slice_dim_name)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        if mesh_impl.tensor_dimension_to_mesh_axis(self._output_dim) is not None:
            raise ValueError("can't pad along split axis")
        inputs = self._inputs[0]
        ndims = self._inputs[0].shape.ndims
        axis = self._axis
        paddings = [[0, 0]] * axis + [self._paddings] + [[0, 0]] * (ndims - axis - 1)

        def slicewise_fn(x, paddings):
            return tf.pad(x, paddings, constant_values=self.pad_value, name="pad")

        y = mesh_impl.slicewise(
            slicewise_fn, lowering.tensors[inputs], paddings)
        lowering.set_tensor_lowering(self.outputs[0], y)


def pad(x, paddings, dim_name, pad_value=0, name=None):
    """Pad operation.
  Args:
    x: a Tensor
    paddings: list of integers of size 2, padding size before and after for dim.
    dim_name: string, name for the padding dim
    name: an optional string
    pad_value: constant value to pad with
  Returns:
    a Tensor
  """
    return CustomPadOperation(
        x, paddings, dim_name, pad_value=pad_value, name=name).outputs[0]

# helpers

def exists(x):
    return x is not None


def get_variable_dtype(bf_16=True):
    # Trainable variable precision
    # Store checkpoints in master type, train in slice type, compute in activation type
    if bf_16:
        return mtf.VariableDType(master_dtype=tf.bfloat16, slice_dtype=tf.float32, activation_dtype=tf.bfloat16)
    else:
        return mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float32, activation_dtype=tf.float32)

def expand_tile(value, newdim, axis=0):
    """Add a new axis of given size."""
    new_shape = value.shape.dims
    new_shape.insert(axis, newdim)
    return mtf.broadcast(value, new_shape)  # shape.dims gets us a list which we need in order to concat
