import pytest
import traceback
import logging
from collections import defaultdict
from contextlib import contextmanager

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import mesh_tensorflow as mtf
from mesh_tensorflow import placement_mesh_impl

from src.dalle_mtf.models import DALLE
from src.dalle_mtf.sample import sample_autoregressive

# helper functions

@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        logging.error(traceback.format_exc())
        raise pytest.fail("DID RAISE {0}".format(exception))

# tests

def test_model():
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    model = DALLE(
        mesh = mesh,
        batch_size = 1,
        n_embd = 16,
        n_heads = 2,
        bf_16 = False
    )

    batch_dim = model.dimensions["batch_dim"]
    sequence_dim = model.dimensions["total_seq_dim"]

    text_inputs = mtf.ones(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32)
    image_inputs = mtf.ones(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32)

    features = {
        'text_inputs': mtf.slice(text_inputs, 0, model.text_seq_len, sequence_dim.name),
        'image_inputs': mtf.slice(image_inputs, 0, model.image_seq_len, sequence_dim.name)
    }

    with not_raises(Exception):
        loss, loss_batch, logits = model.forward(features, return_loss = True, return_logits = True)

        mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        logits = lowering.export_to_tf_tensor(logits)

def test_sampling():
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    model = DALLE(
        mesh = mesh,
        batch_size = 1,
        text_seq_len = 1,
        image_seq_len = 4,
        n_embd = 16,
        n_heads = 2,
        bf_16 = False,
        unique_pad_tokens = True
    )

    batch_dim = model.dimensions["batch_dim"]
    sequence_dim = model.dimensions["total_seq_dim"]

    text_inputs = mtf.ones(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32)
    image_inputs = mtf.zeros(mesh, mtf.Shape((batch_dim, sequence_dim)), tf.int32)

    inputs = {
        'text_inputs': mtf.slice(text_inputs, 0, model.text_seq_len, sequence_dim.name),
        'image_inputs': mtf.slice(image_inputs, 0, model.image_seq_len, sequence_dim.name)
    }

    with not_raises(Exception):
        samples = sample_autoregressive(
            inputs,
            model,
            variable_dtype=mtf.VariableDType(),
            max_steps = sequence_dim.size,
            min_start_pos=model.text_seq_len
        )

        mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        samples = lowering.export_to_tf_tensor(samples)
