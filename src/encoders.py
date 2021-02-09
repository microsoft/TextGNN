import tensorflow as tf
from transformers.modeling_tf_utils import shape_list, get_initializer
try:
    from aggregators import *
except ImportError:
    from .aggregators import *

# simple concat
class SimpleConcat(tf.keras.layers.Layer):
    def __init__(self, config, fanouts, **kwargs):
        super().__init__(**kwargs)
        self.fanouts = fanouts
        self.num_layers = len(self.fanouts) if (len(self.fanouts) > 0 and self.fanouts[0] > 0) else 0
        self.activations = config.gnn_acts

        for l in range(self.num_layers):
            self.dense_layers.append(tf.keras.layers.Dense(
                self.config.hidden_dims[l], activation=self.activations[l], kernel_initializer='glorot_uniform',
                name="dense_%d" % l
            ))

    def call(self, inputs, training=False):
        if self.num_layers == 0:
            return inputs[0]

        hidden = inputs
        for layer in range(self.num_layers):
            next_hidden = []
            for hop in range(self.num_layers - l):
                neighbor = tf.reshape(hidden["bert_" + str(hop+1)], [-1, self.fanouts[hop] * shape_list(hidden["bert_" + str(hop+1)])[-1]])
                seq = tf.concat([hidden["bert_" + str(hop)], neighbor], axis=-1)
                h = self.dense_layers[layer](seq, training=training)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]


# graphsage
class GraphSAGE(tf.keras.layers.Layer):
    def __init__(self, config, fanouts, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.fanouts = fanouts
        self.num_layers = len(self.fanouts) if (len(self.fanouts) > 0 and self.fanouts[0] > 0) else 0
        self.activations = config.gnn_acts

        self.aggregator_class = aggregators.get(self.config.aggregator)
        self.aggs = []
        for layer in range(self.num_layers):
            activation = self.activations[layer]
            if layer == self.num_layers - 1:
                self.aggs.append(
                    self.aggregator_class(self.config, self.config.hidden_dims[layer], activation=activation,
                                          identity_act=True, name='agg_%d' % layer))
            else:
                self.aggs.append(
                    self.aggregator_class(self.config, self.config.hidden_dims[layer], activation=activation,
                                          identity_act=False, name='agg_%d' % layer))

    def call(self, inputs, training=False):
        if self.num_layers == 0:
            return inputs[0]

        dim0 = shape_list(inputs["bert_1"])[-1]
        dims = [dim0] + self.config.hidden_dims[len(self.config.hidden_dims) - self.num_layers:]

        hidden = inputs

        for layer in range(self.num_layers):
            aggregator = self.aggs[layer]
            next_hidden = {}
            for hop in range(self.num_layers - layer):
                neigh_shape = [-1, self.fanouts[hop], dims[layer]]
                h = aggregator((hidden["bert_" + str(hop)], tf.reshape(hidden["bert_" + str(hop+1)], neigh_shape)))
                next_hidden["bert_" + str(hop)] = h
            hidden = next_hidden

        return hidden["bert_0"]


# self attention head
class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, out_size, activation=tf.nn.leaky_relu, residual=False, **kwargs):
        super().__init__(**kwargs)
        self.out_size = out_size
        self.feature_conv = tf.keras.layers.Conv1D(self.out_size, 1, use_bias=False)
        self.f1_conv = tf.keras.layers.Conv1D(1, 1)
        self.f2_conv = tf.keras.layers.Conv1D(1, 1)
        if isinstance(activation, str):
            if activation == "leaky_relu":
                activation = tf.nn.leaky_relu
            elif activation == "relu":
                activation = tf.nn.relu
        self.activation = tf.keras.layers.Activation(activation)
        self.residual = residual

    def build(self, input_shape):
        with tf.name_scope("attn_head"):
            self.bias = self.add_weight(
                "bias", shape=[self.out_size], initializer="zero"
            )
            if self.residual:
                if input_shape[-1] != self.out_size:
                    self.res_conv = tf.keras.layers.Conv1D(self.out_size, 1)
        super().build(self)

    def call(self, seq, training=False):
        seq_fts = self.feature_conv(seq)
        f_1 = self.f1_conv(seq_fts)
        f_2 = self.f2_conv(seq_fts)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        vals = tf.matmul(coefs, seq_fts)
        ret = tf.nn.bias_add(vals, self.bias)

        # residual connection
        if self.residual:
            if shape_list(seq)[-1] != shape_list(ret)[-1]:
                ret = ret + self.res_conv(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


# gat
class GAT(tf.keras.layers.Layer):
    def __init__(self, config, fanouts, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.fanouts = fanouts
        self.num_layers = len(self.fanouts) if (len(self.fanouts) > 0 and self.fanouts[0] > 0) else 0
        self.activations = config.gnn_acts
        self.neighbor_num = self.fanouts[0]
        self.attention_heads = []
        for layer, head_num in enumerate(self.config.head_nums):
            heads = []
            for i in range(head_num):
                heads.append(AttentionHead(self.config.hidden_dims[layer], self.activations[layer], self.config.use_residual))
            self.attention_heads.append(heads)

    def call(self, inputs, training=False):
        if self.num_layers == 0:
            return inputs["bert_0"]

        dim0 = shape_list(inputs["bert_1"])[-1]
        node_feats = tf.expand_dims(inputs["bert_0"], 1)
        neighbor_feats = tf.reshape(inputs["bert_1"], [-1, self.neighbor_num, dim0])
        seq = tf.concat([node_feats, neighbor_feats], 1)

        for layer, head_num in enumerate(self.config.head_nums):
            hidden = []
            for i in range(head_num):
                hidden_val = self.attention_heads[layer][i](seq)
                hidden.append(hidden_val)
            seq = tf.concat(hidden, -1)

        out = hidden
        out = tf.add_n(out) / self.config.head_nums[-1]
        out = tf.slice(out, [0, 0, 0], [-1, 1, self.config.hidden_dims[-1]])
        return tf.reshape(out, [-1, self.config.hidden_dims[-1]])


encoders = {
    'simple': SimpleConcat,
    'graphsage': GraphSAGE,
    'gat': GAT,
}


def get(encoder):
    return encoders.get(encoder)