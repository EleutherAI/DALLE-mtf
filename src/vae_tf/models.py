import math

import tensorflow.compat.v1 as tf
from .layers import gumbel_softmax, mse_loss


# hacked up recompute grad which handles variable scopes properly and to handle bf16
def recompute_grad(f, bf16=False):
    @tf.custom_gradient
    def inner(*args, **kwargs):
        result = tf.stop_gradient(f(*args, **kwargs))
        scope = tf.get_default_graph().get_name_scope()

        def grad(dresult, variables=None):
            with tf.GradientTape() as t:
                t.watch(args)
                if variables is not None:
                    t.watch(variables)
                # we need to outsmart XLA here to force a control dependency
                zero_with_control_dependency = tf.reduce_mean(dresult[0] * 1e-30)
                new_args = []
                for a in args:
                    if a.dtype.is_floating:
                        new_args.append(a + tf.cast(zero_with_control_dependency, a.dtype))
                    else:
                        new_args.append(a)

                with tf.control_dependencies([dresult]):
                    if bf16:
                        with tf.tpu.bfloat16_scope():
                            with tf.variable_scope(scope, reuse=True):
                                result = f(*new_args, **kwargs)
                    else:
                        with tf.variable_scope(scope, reuse=True):
                            result = f(*new_args, **kwargs)
            kw_vars = []
            if variables is not None:
                kw_vars = list(variables)
            grads = t.gradient(result, list(new_args) + kw_vars, output_gradients=[dresult])
            return grads[:len(new_args)], grads[len(new_args):]

        return result, grad
    return inner


class DiscreteVAE:
    def __init__(self,
                 num_tokens,
                 dimensions,
                 convblocks,
                 dim=512,
                 hidden_dim=64,
                 input_channels=3,
                 recompute_grad=False,
                 use_bf16=False,
                 stack_factor=1,
                 ):
        self.num_tokens = num_tokens
        self.dim = dim
        self.hdim = hidden_dim
        self.hdim2 = hidden_dim
        self.num_tokens = num_tokens
        self.num_ch = input_channels
        self.H = self.W = dimensions
        self.height_dim = self.H
        self.width_dim = self.W
        self.conv2d = tf.layers.conv2d
        self.conv2dtranspose = tf.layers.conv2d_transpose
        self.activation = tf.nn.relu
        self.dense = tf.layers.dense
        self.norm = tf.layers.batch_normalization

        # list of (stacked, channels) with implicit stride 2, conv between groups
        self.convblocks = convblocks
        self.recompute_grad = recompute_grad
        self.bf16 = use_bf16

        assert math.log2(stack_factor).is_integer() # maybe you don't actually need this?
        self.stack_factor = stack_factor

    def encoder(self, x):
        if self.bf16:
            x = tf.cast(x, tf.bfloat16)

        if self.stack_factor > 1:
            x = tf.space_to_depth(x, self.stack_factor)

        with tf.variable_scope("encoder"):
            for block, (stack, channels) in enumerate(self.convblocks):
                with tf.variable_scope(f"block_{block}"):
                    for i in range(stack):
                        with tf.variable_scope(f"layer_{i}"):
                            if i == 0:
                                # downsample
                                x = self.conv2d(x, channels, (4, 4), (2, 2), padding="SAME", name=f"conv_downsample")
                            else:
                                # normal residual block

                                def encoder_block(x, channels=channels):
                                    out = self.conv2d(x, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_in")
                                    # out = self.norm(out, name=f"bn_in")
                                    out = self.activation(out, name=f"activ")
                                    out = self.conv2d(out, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_out")
                                    # out = self.norm(out, name=f"bn_out")
                                    return out

                                res_out = recompute_grad(encoder_block, self.bf16)(x) if self.recompute_grad else encoder_block(x)

                                x = x + res_out

        with tf.variable_scope(f"codebook"):
            self.n_hid = x.shape[-1]
            embedding = tf.get_variable("codebook", shape=[self.n_hid, self.num_tokens], dtype=tf.float32)

            if self.bf16:
                x = tf.cast(x, tf.float32)

            output = tf.matmul(x, embedding)

            return output


    def decoder(self, x):
        with tf.variable_scope(f"codebook", reuse=True):
            embedding = tf.get_variable("codebook", shape=[self.n_hid, self.num_tokens], dtype=tf.float32)

            x = tf.matmul(x, embedding, transpose_b=True)

        if self.bf16:
            x = tf.cast(x, tf.bfloat16)

        with tf.variable_scope("decoder"):
            for block, (stack, channels) in enumerate(reversed(self.convblocks)):
                with tf.variable_scope(f"block_{block}"):
                    for i in range(stack):
                        with tf.variable_scope(f"layer_{i}"):
                            if i == 0:
                                # upsample
                                x = self.conv2dtranspose(x, channels, (4, 4), (2, 2), padding="SAME", name=f"conv_upsample")
                            else:
                                # normal residual block

                                def decoder_block(x, channels=channels):
                                    out = self.conv2d(x, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_in")
                                    # out = self.norm(out, name=f"bn_in")
                                    out = self.activation(out, name=f"activ")
                                    out = self.conv2d(out, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_out")
                                    # out = self.norm(out, name=f"bn_out")
                                    return out

                                res_out = recompute_grad(decoder_block, self.bf16)(x) if self.recompute_grad else decoder_block(x)

                                x = x + res_out

            x = self.conv2d(x, self.num_ch * self.stack_factor ** 2, (1, 1), (1, 1))

            if self.bf16:
                x = tf.cast(x, tf.float32)

            if self.stack_factor > 1:
                x = tf.depth_to_space(x, self.stack_factor)

            return x

    def forward(self, features, return_recon_loss=False, return_logits=False, hard_gumbel=True, temperature=1.):
        if isinstance(features, dict):
            img = features["inputs"]
        else:
            img = features
        # NHWC
        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        soft_one_hot = gumbel_softmax(logits, -1, temperature=temperature, hard=hard_gumbel)

        out = self.decoder(soft_one_hot)

        if not return_recon_loss:
            return out

        loss = mse_loss(tf.cast(img, out.dtype), out)
        return loss, out
