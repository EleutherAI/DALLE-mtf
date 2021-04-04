import mesh_tensorflow as mtf
import mesh_tensorflow.transformer as mtf_transformer
import tensorflow.compat.v1 as tf
from math import sqrt
from collections import defaultdict
import math

from .ops import pad, exists, get_variable_dtype, expand_tile, mask_to_bias
from .layers import gumbel_softmax, mse_loss, norm


class DiscreteVAE:
    def __init__(self,
                 num_tokens,
                 batch_size,
                 dimensions,
                 dim=512,
                 hidden_dim=64,
                 input_channels=3,
                 num_layers=3,
                 params=None,
                 bf_16=True):
        self.num_tokens = num_tokens
        self.batch_dim = mtf.Dimension("batch_dim", batch_size)
        self.dim = mtf.Dimension("dim", dim)
        self.hdim = mtf.Dimension("hdim", hidden_dim)
        self.hdim2 = mtf.Dimension("hdim2", hidden_dim)
        self.tokens_dim = mtf.Dimension("tokens_dim", num_tokens)
        self.channels_dim = mtf.Dimension("channels_dim", input_channels)
        self.H = self.W = dimensions
        self.height_dim = mtf.Dimension("height_dim", self.H)
        self.width_dim = mtf.Dimension("width_dim", self.W)
        self.dimensions = {
            "batch_dim": self.batch_dim,
            "dim": self.dim,
            "hdim": self.hdim,
            "hdim2": self.hdim2,
            "tokens_dim": self.tokens_dim,
            "channels_dim": self.channels_dim,
            "height_dim": self.height_dim,
            "width_dim": self.width_dim
        }
        self.conv2d = mtf.layers.conv2d
        self.conv2dtranspose = mtf.layers.conv2d_transpose
        self.activation = mtf.relu
        self.embedding = mtf.layers.embedding
        self.bf_16 = bf_16
        self.variable_dtype = get_variable_dtype(bf_16)
        if params is None:  # extra params
            params = {}
        self.params = defaultdict(lambda: None, params)
        self.num_layers = num_layers

    def encoder_block(self, n, is_last=False):
        def fn(x):
            hdim = self.hdim if n % 2 == 0 else self.hdim2
            x = self.conv2d(x, hdim, (4, 4), (2, 2), padding="SAME", name=f"conv{n}",
                                variable_dtype=self.variable_dtype)
            x = self.activation(x, name=f"enc_activ{n}")
            if is_last:
                x = self.conv2d(x, self.tokens_dim, (1, 1), (1, 1), variable_dtype=self.variable_dtype, name="final_conv")
                tf.logging.info("compressed shape: ")
                tf.logging.info(x.shape)
            return x
        return fn
    
    def decoder_block(self, n, is_last=False):
        def fn(x):
            hdim = self.hdim if n % 2 == 0 else self.hdim2
            x = self.conv2dtranspose(x, hdim, (4, 4), (2, 2), padding="SAME", name=f"convtranspose{n}",
                                        variable_dtype=self.variable_dtype)
            x = self.activation(x, name=f"dec_activ{n}")
            if is_last:
                x = self.conv2d(x, self.channels_dim, (1, 1), (1, 1), variable_dtype=self.variable_dtype, name="final_conv")
            return x
        return fn

    def encoder(self, x):
        with tf.variable_scope("encoder"):
            for n in range(self.num_layers):
                is_last = n == self.num_layers - 1
                block_fn = self.encoder_block(n, is_last)
                if self.params.get("recompute_grad", False) and (self.mode == "train"):
                    x = mtf.recompute_grad(block_fn, [x])
                else:
                    x = block_fn(x)
            return x

    def decoder(self, x):
        with tf.variable_scope("decoder"):
            for n in range(self.num_layers):
                is_last = n == self.num_layers - 1
                block_fn = self.decoder_block(n, is_last)
                if self.params.get("recompute_grad", False) and (self.mode == "train"):
                    x = mtf.recompute_grad(block_fn, [x])
                else:
                    x = block_fn(x)
            return x

    def decode(self, img_seq):
        image_embeds = self.embedding(img_seq, self.tokens_dim, self.dim, variable_dtype=self.variable_dtype,
                                      name="embedding")
        n_dim = image_embeds.shape[1]
        n = n_dim.size
        h = w = int(sqrt(n))
        h_dim = mtf.Dimension("h", h)
        w_dim = mtf.Dimension("w", w)
        out_shape = mtf.Shape([image_embeds.shape[0], image_embeds.shape[-1], h_dim, w_dim])
        image_embeds = mtf.einsum([img_seq], output_shape=out_shape, reduced_dims=[n_dim], name="embd_rearrange")
        images = self.decoder(image_embeds)
        return images

    def forward(self, features, return_recon_loss=False, return_logits=False, hard_gumbel=True):
        if isinstance(features, dict):
            img = features["inputs"]
        else:
            img = features
        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        soft_one_hot = gumbel_softmax(logits, self.tokens_dim, temperature=1., hard=hard_gumbel)
        embedding_weights = mtf.layers.embedding_weights(logits.mesh, self.tokens_dim, self.dim,
                                                         variable_dtype=tf.float32, name="embedding")
        out_shape = mtf.Shape \
            ([soft_one_hot.shape[0], soft_one_hot.shape[1], soft_one_hot.shape[2], embedding_weights.shape[1]])
        sampled = mtf.einsum([soft_one_hot, embedding_weights],
                             output_shape=out_shape, name="codebook_einsum")

        out = self.decoder(sampled)

        denormalize = lambda x: (x + 1) / 2
        if not return_recon_loss:
            return denormalize(out)

        loss = mse_loss(mtf.cast(img, out.dtype), out)
        return loss, denormalize(out)


class DALLE:

    def __init__(self, mesh, n_embd, text_vocab_size=12800, image_vocab_size=512, text_seq_len=256, image_seq_len=1024,
                 n_layers=6, n_heads=8, batch_size=32, bf_16=True, attn_mask=None, mode="train",
                 is_incremental_inference=False, context=None, loss_fn=None, params=None, padding_id=None,
                 activation_fn=None, text_loss_weight=0.15):

        self.mesh = mesh
        self.n_embd = n_embd
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.total_seq_len = text_seq_len + image_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_mask = attn_mask
        self.logits_mask = None
        self.text_loss_weight = text_loss_weight
        self.total_tokens = text_vocab_size + image_vocab_size + 1  # extra for EOS
        self.padding_id = 0 if padding_id is None else padding_id
        self.dimensions = {"embed_dim": mtf.Dimension("embed_dim", n_embd),
                           "text_vocab_dim": mtf.Dimension("vocab_dim", text_vocab_size),
                           "image_vocab_dim": mtf.Dimension("vocab_dim", image_vocab_size),
                           "final_vocab_dim": mtf.Dimension("vocab_dim", self.total_tokens),
                           "text_sequence_dim": mtf.Dimension("sequence_dim", text_seq_len),
                           "image_sequence_dim": mtf.Dimension("sequence_dim", image_seq_len),
                           "total_seq_dim": mtf.Dimension("sequence_dim", self.total_seq_len),
                           "embed_seq_dim": mtf.Dimension("embed_seq_dim", self.total_seq_len),
                           "memory_len_dim": mtf.Dimension("memory_len_dim", self.total_seq_len),
                           "heads_dim": mtf.Dimension("heads", n_heads),
                           "kv_dim": mtf.Dimension("kv_dim", n_embd // n_heads),
                           "batch_dim": mtf.Dimension("batch_dim", batch_size)}
        self.bf_16 = bf_16
        self.variable_dtype = get_variable_dtype(bf_16)
        self.mode = mode
        self.context = context
        self.is_incremental_inference = is_incremental_inference
        if loss_fn is None:
            loss_fn = mtf.layers.softmax_cross_entropy_with_logits
        self.loss_fn = loss_fn
        if activation_fn is None:
            activation_fn = mtf.relu
        self.activation_fn = activation_fn
        if self.is_incremental_inference:
            assert self.context is not None, "must have context in incremental inference"
            assert self.context['mode'] == 'incremental'
        if params is None:  # extra params
            params = {}
        self.params = defaultdict(lambda: None, params)

    def embedding(self, x, name):
        embd_dim = self.dimensions["embed_dim"]
        if "text" in name:
            vocab_dim = self.dimensions["text_vocab_dim"]
        else:
            vocab_dim = self.dimensions["image_vocab_dim"]
        with tf.variable_scope(name):
            wte = mtf.get_variable(x.mesh, "wte",
                                   mtf.Shape([vocab_dim, embd_dim]),
                                   initializer=tf.random_normal_initializer(stddev=0.02),
                                   master_dtype=self.variable_dtype.master_dtype,
                                   slice_dtype=self.variable_dtype.slice_dtype,
                                   activation_dtype=self.variable_dtype.activation_dtype)
            # Text embedding
            x = mtf.gather(wte, x, vocab_dim)
            embed_dropout = self.params.get("embed_dropout", 0)
            if embed_dropout > 0 and self.mode == "train":
                x = mtf.dropout(x, rate=embed_dropout, name="wte_dropout")
        return x

    def positional_embedding(self, x, name):
        if "text" in name:
            sequence_dim = self.dimensions["text_sequence_dim"]
        else:
            sequence_dim = self.dimensions["image_sequence_dim"]
        with tf.variable_scope(name):
            # Positional embedding
            wpe = mtf.get_variable(x.mesh, "wpe",
                                   mtf.Shape([self.dimensions["embed_seq_dim"], self.dimensions["embed_dim"]]),
                                   initializer=tf.random_normal_initializer(stddev=0.01),
                                   master_dtype=self.variable_dtype.master_dtype,
                                   slice_dtype=self.variable_dtype.slice_dtype,
                                   activation_dtype=self.variable_dtype.activation_dtype)
            position_indices = mtf.range(x.mesh, sequence_dim, tf.int64) if not \
                self.is_incremental_inference else (self.context.position - 1)
            pos_emb = mtf.gather(wpe, position_indices, wpe.shape[0])
            embed_dropout = self.params.get("embed_dropout", 0)
            if embed_dropout > 0 and self.mode == "train":
                pos_emb = mtf.dropout(pos_emb, rate=embed_dropout, name="wte_dropout")
            x += pos_emb
            return x

    def get_attn_mask(self, mesh, nd, ns):
        if not exists(self.attn_mask):
            i = mtf.range(mesh, nd, tf.int32) + ns.size - nd.size
            j = mtf.range(mesh, ns, tf.int32)
            i, j = map(lambda t: mtf.broadcast(t, [nd, ns]), (i, j))
            self.attn_mask = mtf.cast(mtf.less(i, j), self.variable_dtype.activation_dtype) * -1e10
        return self.attn_mask
    
    def set_logits_mask(self, tf_mask):
        mask_shape = mtf.Shape([self.dimensions['total_seq_dim'], self.dimensions['final_vocab_dim']])
        mtf_mask = mtf.import_fully_replicated(self.mesh, tf_mask, mask_shape)
        new_shape = mtf.Shape([self.dimensions['batch_dim'], self.dimensions['total_seq_dim'], self.dimensions['final_vocab_dim']])
        mtf_mask = mtf.broadcast(mtf_mask, new_shape)
        self.logits_mask = mtf_mask

    def attention(self, x, n_state, mask, attention_type="global", name="attn"):
        if not self.is_incremental_inference:
            # x :: [batch, seq, n_embd]
            batch_dim, seq_dim, embd_dim = x_shape = x.shape
        else:
            batch_dim, embd_dim = x_shape = x.shape
            seq_dim = self.dimensions['total_seq_dim']

        assert n_state.size % self.n_heads == 0, "n_state must be divisible by n_heads"
        with tf.variable_scope(name):
            # Compute attention inputs
            mtfparams = mtf.transformer.attention.attention_params_simple(
                x.mesh,
                io_dim=self.dimensions["embed_dim"],
                kv_dim=self.dimensions["kv_dim"],
                heads_dim=self.dimensions["heads_dim"],
                variable_dtype=self.variable_dtype
            )
            q = mtfparams.compute_q(x)
            k = mtfparams.compute_k(x)
            v = mtfparams.compute_v(x)

            if self.is_incremental_inference:
                one_hot = mtf.one_hot(self.context.position - 1, seq_dim, dtype=self.variable_dtype.master_dtype)
                inv_one_hot = 1.0 - one_hot
                old_k, old_v = self.context.get_states(2)
                k = old_k * inv_one_hot + k * one_hot
                v = old_v * inv_one_hot + v * one_hot

            if exists(self.context):
                self.context.record_new_states([k, v])

            with tf.variable_scope("attention"):
                if attention_type == "global":
                    if exists(mask):
                        if not self.is_incremental_inference:
                            broadcasted_mask = mtf.broadcast(mask,
                                                             [batch_dim, self.dimensions["heads_dim"], mask.shape[-2],
                                                              mask.shape[-1]])  # TODO: not sure this is correct
                        else:
                            # In the incremental case, a custom mask needs to be built that masks out all key/values that are greater than the current position
                            mask = mtf.gather(mask, self.context.position - 1, seq_dim)
                            broadcasted_mask = mtf.broadcast(mask,
                                                             [batch_dim, self.dimensions["heads_dim"], mask.shape[-1]])

                    k = mtf.replace_dimensions(k, k.shape[1], self.dimensions["memory_len_dim"])
                    v = mtf.replace_dimensions(v, v.shape[1], self.dimensions["memory_len_dim"])

                    attn_dropout_rate = self.params.get("attention_dropout", 0) if self.mode == "train" else 0

                    a = mtf_transformer.attention.attention(
                        q, k, v,
                        memory_length_dim=self.dimensions["memory_len_dim"],
                        key_dim=self.dimensions["kv_dim"],
                        value_dim=self.dimensions["kv_dim"],
                        bias=broadcasted_mask,
                        dropout_rate=attn_dropout_rate
                    )
                else:
                    raise NotImplementedError("Unknown attention type {}!".format(attention_type))

            with tf.variable_scope("compute_output"):
                a = mtfparams.compute_output(a, x_shape)

            with tf.variable_scope("compute_output_bias"):
                b = mtf.get_variable(x.mesh, "o_b", [embd_dim], initializer=tf.constant_initializer(0),
                                     master_dtype=self.variable_dtype.master_dtype,
                                     slice_dtype=self.variable_dtype.slice_dtype,
                                     activation_dtype=self.variable_dtype.activation_dtype)
                a += b
            residual_dropout = self.params.get("residual_dropout", 0)
            if self.mode == "train" and residual_dropout > 0:
                a = mtf.dropout(a, rate=residual_dropout, name="res_dropout")
            return a

    def mlp(self, x, n_state, name="mlp"):
        residual_dropout = self.params.get("residual_dropout", 0)
        with tf.variable_scope(name):
            h = self.activation_fn(self.linear(x, n_state, name="mlp_linear_1"))
            h2 = self.linear(h, x.shape[-1], name="mlp_linear_2", scale=True)
            if self.mode == "train" and residual_dropout > 0:
                h2 = mtf.dropout(h2, rate=residual_dropout, name="mlp_dropout")
            return h2

    def block(self, mask, name):
        def fn(x):
            with tf.variable_scope(name):
                intermediate_size = x.shape[-1].size * 4  # Grab last dimension from input
                x += self.attention(self.layer_norm(x, "norm_1"), x.shape[-1], name="attn", mask=mask)
                # Define intermediate layer of mlp - to split
                dim_intermediate_expanded = mtf.Dimension("intermediate_expanded", intermediate_size)
                x += self.mlp(self.layer_norm(x, "norm_2"), dim_intermediate_expanded, name="mlp")
                return x
        return fn

    def transformer(self, x, mask):
        for layer in range(self.n_layers):
            # attn blocks
            block_fn = self.block(mask, f"layer_{layer}")
            # If true and in train mode, enable gradient checkpointing
            if self.params.get("recompute_grad", False) and (self.mode == "train"):
                x = mtf.recompute_grad(block_fn, [x])
            else:
                x = block_fn(x)
        return x

    def _loss(self, logits, labels):
        with tf.variable_scope("loss_final"):
            loss_batch = self.loss_fn(logits =mtf.slice(logits, begin=self.text_seq_len, size=self.image_seq_len, slice_dim_name="sequence_dim"), 
                                      targets=mtf.slice(labels, begin=self.text_seq_len, size=self.image_seq_len, slice_dim_name="sequence_dim"),
                                      vocab_dim=logits.shape[-1], z_loss=0.0)

        with tf.variable_scope("reduce_mean_final"):
            loss = mtf.reduce_mean(loss_batch)

        loss /= self.params.get("num_microbatches", 1)
        # Convert to train dtype
        loss = mtf.cast(loss, self.variable_dtype.slice_dtype)
        return loss, loss_batch  # loss batch must be returned for metric fns

    def linear(self, x, new_dim, w_init_stdev=0.02, params=None, scale=False, name="linear"):
        # nf = number of features
        scale_type = self.params.get("scale_type", "scale_by_depth")
        if scale_type == "scale_by_depth" and scale:
            # Scale by sqrt(num_layers), only happens at the final projection before a res block output
            w_init_stdev = w_init_stdev * (1. / math.sqrt(self.n_layers))
        if scale_type == "scale_by_in":  # Scale by sqrt(num_input_features)
            w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].size))
        return mtf.layers.dense(x, new_dims=[new_dim], reduced_dims=[x.shape[-1]], name=name, use_bias=True,
                                kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                                variable_dtype=self.variable_dtype)

    def axial_positional_embedding(self, mesh, name):
        with tf.variable_scope(name):
            axial_dim_side = int(sqrt(self.image_seq_len))

            embd_dim = self.dimensions["embed_dim"]
            axial_dim = mtf.Dimension("axial_dim", self.image_seq_len)

            dim_axials = [mtf.Dimension(f"axial_dim_{i}", t) for i, t in enumerate((axial_dim_side, axial_dim_side))]

            axial_wpe_1 = mtf.get_variable(mesh, "axial_wpe_1", mtf.Shape([dim_axials[0], embd_dim]),
                                           initializer=tf.random_normal_initializer(stddev=0.01),
                                           master_dtype=self.variable_dtype.master_dtype,
                                           slice_dtype=self.variable_dtype.slice_dtype,
                                           activation_dtype=self.variable_dtype.activation_dtype)

            axial_wpe_2 = mtf.get_variable(mesh, "axial_wpe_2", mtf.Shape([dim_axials[1], embd_dim]),
                                           initializer=tf.random_normal_initializer(stddev=0.01),
                                           master_dtype=self.variable_dtype.master_dtype,
                                           slice_dtype=self.variable_dtype.slice_dtype,
                                           activation_dtype=self.variable_dtype.activation_dtype)

            axial_wpe_1, axial_wpe_2 = map(lambda t: mtf.broadcast(t, [dim_axials[0], dim_axials[1], embd_dim]),
                                           (axial_wpe_1, axial_wpe_2))
            wpe = (axial_wpe_1 + axial_wpe_2) / 2

            wpe = mtf.reshape(wpe, [axial_dim, embd_dim])
            wpe = pad(wpe, [self.text_seq_len, 0], axial_dim.name)
            wpe = mtf.replace_dimensions(wpe, wpe.shape[0], self.dimensions["embed_seq_dim"])
            return wpe


    def absolute_positional_embedding(self, mesh, name):
        with tf.variable_scope(name):
            # Positional embedding
            wpe = mtf.get_variable(mesh, "wpe",
                                   mtf.Shape([self.dimensions["embed_seq_dim"], self.dimensions["embed_dim"]]),
                                   initializer=tf.random_normal_initializer(stddev=0.01),
                                   master_dtype=self.variable_dtype.master_dtype,
                                   slice_dtype=self.variable_dtype.slice_dtype,
                                   activation_dtype=self.variable_dtype.activation_dtype)
            return wpe

    def apply_positional_embedding(self, x, wpe):
        position_indices = mtf.range(x.mesh, self.dimensions["total_seq_dim"], tf.int64) if not \
            self.is_incremental_inference else (self.context.position - 1)
        pos_emb = mtf.gather(wpe, position_indices, wpe.shape[0])
        embed_dropout = self.params.get("embed_dropout", 0)
        if embed_dropout > 0 and self.mode == "train":
            pos_emb = mtf.dropout(pos_emb, rate=embed_dropout, name="wte_dropout")
        x += pos_emb
        return x

    def layer_norm(self, x, name="layer_norm", axis=None, epsilon=1e-5):
        """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
        if axis is None:
            axis = x.shape[-1]
        with tf.variable_scope(name):
            g = mtf.get_variable(x.mesh, "g", [axis], initializer=tf.constant_initializer(1),
                                 master_dtype=self.variable_dtype.master_dtype,
                                 slice_dtype=self.variable_dtype.slice_dtype,
                                 activation_dtype=self.variable_dtype.activation_dtype)
            b = mtf.get_variable(x.mesh, "b", [axis], initializer=tf.constant_initializer(0),
                                 master_dtype=self.variable_dtype.master_dtype,
                                 slice_dtype=self.variable_dtype.slice_dtype,
                                 activation_dtype=self.variable_dtype.activation_dtype)

            x = norm(x, axis, epsilon)
            x = x * g + b
            return x

    def to_logits(self, x):
        with tf.variable_scope("to_logits"):
            logits = self.linear(self.layer_norm(x), self.dimensions["final_vocab_dim"], name="linear_out")
            # Go to full precision for the logits
            if self.is_incremental_inference:
                # add seq dim in inference mode
                logits = expand_tile(logits, mtf.Dimension("sequence_dim", 1), axis=1)
            return mtf.cast(logits, tf.float32)

    def to_image_logits(self, x):
        with tf.variable_scope("to_logits"):
            if not self.is_incremental_inference:
                x = mtf.slice(x, begin = self.text_seq_len, size = self.image_seq_len, slice_dim_name = x.shape[1].name)

            image_logits = self.linear(x, self.dimensions["image_vocab_dim"], name="linear_image_out")

            # Go to full precision for the logits
            image_logits = mtf.cast(image_logits, tf.float32)
            return image_logits

    def to_text_logits(self, x):
        with tf.variable_scope("to_logits"):
            text_tokens = mtf.slice(x, begin = 0, size = self.text_seq_len, slice_dim_name = x.shape[1].name)
            text_logits = self.linear(text_tokens, self.dimensions["text_vocab_dim"], name="linear_text_out")

            # Go to full precision for the logits
            text_logits = mtf.cast(text_logits, tf.float32)
            return text_logits

    def _loss(self, text_logits, image_logits, text_labels, image_labels):
        with tf.variable_scope("loss_final"):
            text_loss_batch = self.loss_fn(logits=text_logits, targets=text_labels,
                                      vocab_dim=text_logits.shape[-1], z_loss=0.0)

            image_loss_batch = self.loss_fn(logits=image_logits, targets=image_labels,
                                      vocab_dim=image_logits.shape[-1], z_loss=0.0)

            loss_batch = text_loss_batch * self.text_loss_weight + image_loss_batch

        with tf.variable_scope("reduce_mean_final"):
            loss = mtf.reduce_mean(loss_batch)

        loss /= self.params.get("num_microbatches", 1)
        # Convert to train dtype
        loss = mtf.cast(loss, self.variable_dtype.slice_dtype)
        return loss, loss_batch  # loss batch must be returned for metric fns

    def forward(self, features, return_loss=True, return_logits=False):
        if features.get('text_inputs') is not None:
            text = features["text_inputs"]
            text_with_bos = pad(text, [1, 0], dim_name = text.shape[1].name, pad_value = self.padding_id)
            text_emb = self.embedding(text_with_bos, "text_embd")
        else:
            assert self.is_incremental_inference

        image = features.get("image_inputs", None)

        if not self.is_incremental_inference:
            image_input = mtf.slice(image, 0, self.image_seq_len - 1, image.shape[1].name)
            image_emb = self.embedding(image_input, "image_embd")
            tokens = mtf.concat([text_emb, image_emb], concat_dim_name="sequence_dim")  # [batch, seq, n_embd]
        else:
            # reshape inputs if in inference mode
            image = mtf.gather(image, self.context.position - 1, self.dimensions["image_sequence_dim"])
            image = mtf.reshape(image, [self.dimensions["batch_dim"]])
            tokens = self.embedding(image, "image_embd")

        # positional embedding

        abs_pos_emb = self.absolute_positional_embedding(tokens.mesh, "positional_embedding")
        axial_pos_emb = self.axial_positional_embedding(tokens.mesh, "axial_positional_embedding")

        tokens = self.apply_positional_embedding(tokens, abs_pos_emb)
        tokens = self.apply_positional_embedding(tokens, axial_pos_emb)

        # attention

        mask = self.get_attn_mask(tokens.mesh, self.dimensions["total_seq_dim"], self.dimensions["memory_len_dim"])
        out = self.transformer(tokens, mask=mask)

        # to logits

        image_logits = self.to_image_logits(out)

        if not return_loss:
            logits = mtf.cast(image_logits, self.variable_dtype.master_dtype)
            return logits

        assert exists(image), 'when training, image must be supplied'
        labels = mtf.concat([text, image], concat_dim_name="sequence_dim")

        text_logits = self.to_text_logits(out)

        text_labels = mtf.slice(labels, begin = 0, size = self.text_seq_len, slice_dim_name = labels.shape[1].name)
        image_labels = mtf.slice(labels, begin = self.text_seq_len, size = self.image_seq_len, slice_dim_name = labels.shape[1].name)

        loss, loss_batch = self._loss(text_logits, image_logits, text_labels, image_labels)

        if return_logits and return_loss:
            # Cast back to checkpoint dtype
            logits = mtf.cast(image_logits, self.variable_dtype.master_dtype)
            return loss, loss_batch, logits
        return loss, loss_batch
