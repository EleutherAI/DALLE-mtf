import tensorflow.compat.v1 as tf
from .layers import gumbel_softmax, mse_loss

class DiscreteVAE:
    def __init__(self,
                 num_tokens,
                 dimensions,
                 convblocks,
                 dim=512,
                 hidden_dim=64,
                 input_channels=3,
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

    def encoder(self, x):
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
                                out = self.conv2d(x, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_in")
                                # out = self.norm(out, name=f"bn_in")
                                out = self.activation(out, name=f"activ")
                                out = self.conv2d(out, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_out")
                                # out = self.norm(out, name=f"bn_out")

                                x = x + out

        with tf.variable_scope(f"codebook"):
            self.n_hid = x.shape[-1]
            embedding = tf.get_variable("codebook", shape=[self.n_hid, self.num_tokens], dtype=tf.float32)

            return tf.matmul(x, embedding)

    def decoder(self, x):
        with tf.variable_scope(f"codebook", reuse=True):
            embedding = tf.get_variable("codebook", shape=[self.n_hid, self.num_tokens], dtype=tf.float32)

            x = tf.matmul(x, embedding, transpose_b=True)

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
                                out = self.conv2d(x, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_in")
                                # out = self.norm(out, name=f"bn_in")
                                out = self.activation(out, name=f"activ")
                                out = self.conv2d(out, channels, (3, 3), (1, 1), padding="SAME", name=f"conv_out")
                                # out = self.norm(out, name=f"bn_out")

                                x = x + out

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

        out = self.decoder(soft_one_hot)

        denormalize = lambda x: (x + 1) / 2
        if not return_recon_loss:
            return denormalize(out)

        loss = mse_loss(tf.cast(img, out.dtype), out)
        return loss, denormalize(out)
