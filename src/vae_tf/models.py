import tensorflow.compat.v1 as tf
from math import sqrt
from collections import defaultdict
import math

from .ops import pad, exists, get_variable_dtype
from .layers import gumbel_softmax, mse_loss

class DiscreteVAE:
    def __init__(self,
                 num_tokens,
                 dimensions,
                 dim=512,
                 hidden_dim=64,
                 input_channels=3,
                 bf_16=True):
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
        self.bf_16 = bf_16
        self.num_layers = 3  # hardcode this for now

    def encoder(self, x):
        with tf.variable_scope("encoder"):
            x = self.conv2d(x, self.hdim, (4, 4), (2, 2), padding="SAME", name="conv1")
            x = self.activation(x, name="activ1")
            x = self.conv2d(x, self.hdim2, (4, 4), (2, 2), padding="SAME", name="conv2")
            x = self.activation(x, name="activ2")
            x = self.conv2d(x, self.hdim, (4, 4), (2, 2), padding="SAME", name="conv3")
            x = self.activation(x, name="activ3")
            x = self.conv2d(x, self.num_tokens, (1, 1), (1, 1))
            return x

    def decoder(self, x):
        with tf.variable_scope("decoder"):
            x = self.conv2dtranspose(x, self.hdim2, (4, 4), (2, 2), padding="SAME", name="convtranspose1")
            x = self.activation(x, name="activ1")
            x = self.conv2dtranspose(x, self.hdim, (4, 4), (2, 2), padding="SAME", name="convtranspose2")
            x = self.activation(x, name="activ2")
            x = self.conv2dtranspose(x, self.hdim2, (4, 4), (2, 2), padding="SAME", name="convtranspose3")
            x = self.activation(x, name="activ3")
            x = self.conv2d(x, self.num_ch, (1, 1), (1, 1))
            return x

    def forward(self, features, return_recon_loss=False, return_logits=False, hard_gumbel=True):
        if isinstance(features, dict):
            img = features["inputs"]
        else:
            img = features
        # NHWC
        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        soft_one_hot = gumbel_softmax(logits, -1, temperature=1., hard=hard_gumbel)
        sampled = self.dense(soft_one_hot, self.hdim)

        out = self.decoder(sampled)

        denormalize = lambda x: (x + 1) / 2
        if not return_recon_loss:
            return denormalize(out)

        loss = mse_loss(tf.cast(img, out.dtype), out)
        return loss, denormalize(out)
