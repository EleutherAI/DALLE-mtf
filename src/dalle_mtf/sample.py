import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import mesh_tensorflow.transformer as mtf_transformer


def sample_autoregressive(inputs,
                          model,
                          max_steps=None,
                          temperature=0.9,
                          padding_id = 0,
                          variable_dtype=mtf.VariableDType(tf.float32),
                          sampling_keep_top_k=-1,
                          cached=True,
                          min_start_pos=None
                          ):
    """Sample randomly one token at a time.

    The partial_sequences represent partial sequences to be continued.  The
    first tokens of each sequence are nonzero representing the given partial
    sequences and the last tokens of each sequence are zeros, representing what
    needs to be filled in.

    If there are no partial sequences (you want to sample from the beginning),
    then pass partial_sequences=mtf.zeros(mesh, shape, dtype=tf.int32) and

    Args:
        inputs: an input dictionary containing 'text_inputs' and 'image_inputs',
        model: DALL-E model
        stop_at_token: an optional integer eos id.  Stop when we produce it.
        max_steps: an optional integer, the max number of steps to decode.
        temperature: an optional floating point value between 0.0 and 1.0 0.0
        means argmax, 1.0 means sample according to predicted distribution.
        variable_dtype: a mtf.VariableDType
        decoding, one per each input layer + the embedding layer
        sampling_keep_top_k: an integer - if not -1, only sample from the top k
        logits.

    Returns:
        a Tensor with shape [<batch_dims>, length_dim]
    """

    # with dalle, inputs will be a text sequence of len 256, then the rest image tokens.
    # the parts we want to fill in will be <|pad_token|>, which we should assign in the input

    batch_dims = model.dimensions["batch_dim"]
    length_dim = model.dimensions["total_seq_dim"]
    image_seq_dim = model.dimensions['image_sequence_dim']

    image_inputs = inputs['image_inputs']
    text_inputs = inputs['text_inputs']

    # Gets position (in image inputs) where zero padding starts
    initial_position = mtf.reduce_sum(
        mtf.to_int32(mtf.not_equal(image_inputs, padding_id)),
        reduced_dim=image_seq_dim)

    if min_start_pos is not None:
        # force the sampling to never start below a minimum starting position, say the text length.
        # this will also be useful for image completion, where you can start sampling from half the image tokens
        initial_position = mtf.maximum(initial_position, min_start_pos)

    # initial_position += model.dimensions['text_seq_dim'].size

    length_range = mtf.range(image_inputs.mesh, image_seq_dim, tf.int32)

    # Builds context to pass around internally
    # The 'first part' context records initial states of k / v / x
    if cached:
        context_first_part = mtf_transformer.transformer.Context(
            model=None,
            mesh=image_inputs.mesh,
            batch_dims=batch_dims,
            length_dim=image_seq_dim,
            variable_dtype=variable_dtype,
            mode="first_part",
            position=length_range,
            position_is_default=True,
            new_states=[],
            initial_position=initial_position,
            sequence_id=None,
            constant_states=[],
            inputs=inputs)
        model.context = context_first_part

        with tf.variable_scope('dall-e'):
            logits = model.forward(inputs, return_loss=False, return_logits=True)
        del logits

        initial_states = context_first_part.new_states
    else:
        initial_states = []

    def cond_fn(position, ids, *unused_states):
        """Should we run another loop iteration?"""
        past_end = mtf.greater_equal(position, image_seq_dim.size)
        if max_steps:
            past_end = mtf.logical_or(
                past_end, mtf.greater_equal(position - initial_position, max_steps))

        is_done = past_end
        all_done = mtf.reduce_all(is_done)
        return mtf.logical_not(all_done)

    def body_fn(position, ids, *states):
        """One step in the decode loop."""
        nonlocal sampling_keep_top_k

        context = mtf_transformer.transformer.Context(
            model=None,
            mesh=image_inputs.mesh,
            batch_dims=batch_dims,
            length_dim=image_seq_dim,
            variable_dtype=variable_dtype,
            mode="incremental",
            position=position,
            position_is_default=True,
            states=states,
            new_states=[],
            initial_position=position,
            sequence_id=None,
            inputs=ids) if cached else None

        model.is_incremental_inference = True if cached else False
        model.context = context
        with tf.variable_scope("dall-e", reuse=tf.AUTO_REUSE):
            logits = model.forward({'image_inputs': image_inputs, 'text_inputs': (text_inputs if not cached else None)}, return_loss=False, return_logits=True)

        # By default, do top_k sampling of 0.9
        if sampling_keep_top_k == -2:
            sampling_keep_top_k = int(logits.shape[-1].size * 0.1)

        if sampling_keep_top_k != -1:
            if sampling_keep_top_k <= 0:
                raise ValueError("sampling_keep_top_k must either be -1 or positive.")
            k_largest = mtf.nth_largest_element(
                logits, n=sampling_keep_top_k,
                reduced_dim=model.dimensions['image_vocab_dim'])
            logits = mtf.where(mtf.less_equal(logits, k_largest),
                               mtf.ones_like(logits) * -1e6, logits)

        # temperature sampling
        ids_this_step = mtf.sample_with_temperature(
            logits, model.dimensions['image_vocab_dim'], temperature)

        # reshape & assign results
        if cached:
            ids_this_step = mtf.reshape(ids_this_step, ([batch_dims]))
        else:
            ids_this_step = mtf.shift(ids_this_step, offset=1, dim=image_seq_dim, wrap=False)

        one_hot = mtf.one_hot(position, image_seq_dim, dtype=tf.int32)
        one_new_id = ids_this_step * one_hot
        new_ids = (1 - one_hot) * ids + one_new_id
        new_position = position + 1
        ret = [new_position, new_ids]

        if cached:
            ret += context.new_states
        return ret

    while_loop_inputs = [initial_position, image_inputs] + initial_states
    final_position, outputs = mtf.while_loop(
        cond_fn, body_fn, while_loop_inputs)[:2]
    del final_position
    return outputs
