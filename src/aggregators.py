## GraphSage aggregators
import logging

import numpy as np
import tensorflow as tf
from transformers.modeling_tf_utils import shape_list

logger = logging.getLogger(__name__)


class MeanAggregator(tf.keras.layers.Layer):

    def __init__(self, config, out_dim, activation='relu', identity_act=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)
        self.concat = config.agg_concat
        self.add_bias = config.agg_bias
        self.out_dim = out_dim
        self.identity_act = identity_act

        self.self_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="self_weight"
        )
        self.neigh_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="neigh_weights"
        )

        if not self.identity_act:
            self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        if self.add_bias:
            with tf.name_scope('bias'):
                self.bias = self.add_weight(
                    "weight",
                    shape=[self.out_dim * 2 if self.concat else self.out_dim],
                    initializer='zeros',
                )
        super().build(input_shape)

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = self.dropout(neigh_vecs, training=training)
        self_vecs = self.dropout(self_vecs, training=training)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        from_neighs = self.neigh_weights(neigh_means)
        from_self = self.self_weights(self_vecs)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.add_bias:
            output += self.bias

        if self.identity_act: return output
        return self.act(output)


class GCNAggregator(tf.keras.layers.Layer):

    def __init__(self, config, out_dim, activation='relu', identity_act=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)
        self.out_dim = out_dim

        if identity_act:
            self.neigh_weights = tf.keras.layers.Dense(
                self.out_dim, use_bias=config.agg_bias, kernel_initializer='glorot_uniform',
                name="neigh_weights"
            )
        else:
            self.neigh_weights = tf.keras.layers.Dense(
                self.out_dim, use_bias=config.agg_bias, activation=activation, kernel_initializer='glorot_uniform',
                name="neigh_weights"
            )

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = self.dropout(neigh_vecs, training=training)
        self_vecs = self.dropout(self_vecs, training=training)
        means = tf.reduce_mean(tf.concat([neigh_vecs,
                                          tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        return self.neigh_weights(means)


class MaxPoolingAggregator(tf.keras.layers.Layer):

    def __init__(self, config, out_dim, activation='relu', identity_act=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)
        self.concat = config.agg_concat
        self.add_bias = config.agg_bias
        self.out_dim = out_dim
        self.identity_act = identity_act
        self.hidden_dim = 512 if config.agg_model_size == 'small' else 1024

        self.mlp_layers = []
        self.mlp_layers.append(tf.keras.layers.Dense(
            self.hidden_dim, activation='relu', kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(config.weight_decay), name="neigh_mlp"
        ))

        self.self_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="self_weight"
        )
        self.neigh_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="neigh_weights"
        )

        if not self.identity_act:
            self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        if self.add_bias:
            with tf.name_scope('bias'):
                self.bias = self.add_weight(
                    "weight",
                    shape=[self.out_dim * 2 if self.concat else self.out_dim],
                    initializer='zeros',
                )
        super().build(input_shape)

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs = inputs

        for l in self.mlp_layers:
            neigh_vecs = self.dropout(neigh_vecs, training=training)
            neigh_vecs = l(neigh_vecs)
        neigh_vecs = tf.reduce_max(neigh_vecs, axis=1)

        from_neighs = self.neigh_weights(neigh_vecs)
        from_self = self.self_weights(self_vecs)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.add_bias:
            output += self.bias

        if self.identity_act: return output
        return self.act(output)


class MeanPoolingAggregator(tf.keras.layers.Layer):

    def __init__(self, config, out_dim, activation='relu', identity_act=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)
        self.concat = config.agg_concat
        self.add_bias = config.agg_bias
        self.out_dim = out_dim
        self.identity_act = identity_act
        self.hidden_dim = 512 if config.agg_model_size == 'small' else 1024

        self.mlp_layers = []
        self.mlp_layers.append(tf.keras.layers.Dense(
            self.hidden_dim, activation='relu', kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(config.weight_decay), name="neigh_mlp"
        ))

        self.self_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="self_weight"
        )
        self.neigh_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="neigh_weights"
        )

        if not self.identity_act:
            self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        if self.add_bias:
            with tf.name_scope('bias'):
                self.bias = self.add_weight(
                    "weight",
                    shape=[self.out_dim * 2 if self.concat else self.out_dim],
                    initializer='zeros',
                )
        super().build(input_shape)

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs = inputs

        for l in self.mlp_layers:
            neigh_vecs = self.dropout(neigh_vecs, training=training)
            neigh_vecs = l(neigh_vecs)
        neigh_vecs = tf.reduce_mean(neigh_vecs, axis=1)

        from_neighs = self.neigh_weights(neigh_vecs)
        from_self = self.self_weights(self_vecs)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.add_bias:
            output += self.bias

        if self.identity_act: return output
        return self.act(output)


class TwoMaxLayerPoolingAggregator(tf.keras.layers.Layer):

    def __init__(self, config, out_dim, activation='relu', identity_act=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)
        self.concat = config.agg_concat
        self.add_bias = config.agg_bias
        self.out_dim = out_dim
        self.identity_act = identity_act
        self.hidden_dim_1 = 512 if config.agg_model_size == 'small' else 1024
        self.hidden_dim_1 = 256 if config.agg_model_size == 'small' else 512

        self.mlp_layers = []
        self.mlp_layers.append(tf.keras.layers.Dense(
            self.hidden_dim1, activation='relu', kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(config.weight_decay), name="neigh_mlp_1"
        ))
        self.mlp_layers.append(tf.keras.layers.Dense(
            self.hidden_dim2, activation='relu', kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(config.weight_decay), name="neigh_mlp_2"
        ))

        self.self_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="self_weight"
        )
        self.neigh_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="neigh_weights"
        )

        if not self.identity_act:
            self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        if self.add_bias:
            with tf.name_scope('bias'):
                self.bias = self.add_weight(
                    "weight",
                    shape=[self.out_dim * 2 if self.concat else self.out_dim],
                    initializer='zeros',
                )
        super().build(input_shape)

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs = inputs

        for l in self.mlp_layers:
            neigh_vecs = self.dropout(neigh_vecs, training=training)
            neigh_vecs = l(neigh_vecs)
        neigh_vecs = tf.reduce_max(neigh_vecs, axis=1)

        from_neighs = self.neigh_weights(neigh_vecs)
        from_self = self.self_weights(self_vecs)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.add_bias:
            output += self.bias

        if self.identity_act: return output
        return self.act(output)


class SeqAggregator(tf.keras.layers.Layer):

    def __init__(self, config, out_dim, activation='relu', identity_act=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)
        self.concat = config.agg_concat
        self.add_bias = config.agg_bias
        self.out_dim = out_dim
        self.identity_act = identity_act
        self.hidden_dim = 128 if config.agg_model_size == 'small' else 256

        self.lstm = tf.keras.layers.LSTM(self.hidden_dim)

        self.self_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="self_weight"
        )
        self.neigh_weights = tf.keras.layers.Dense(
            self.out_dim, use_bias=False, kernel_initializer='glorot_uniform', name="neigh_weights"
        )

        if not self.identity_act:
            self.act = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        if self.add_bias:
            with tf.name_scope('bias'):
                self.bias = self.add_weight(
                    "weight",
                    shape=[self.out_dim * 2 if self.concat else self.out_dim],
                    initializer='zeros',
                )
        super().build(input_shape)

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs = inputs

        mask = tf.cast(tf.sign(tf.reduce_max(tf.abs(x), axis=2)), dtype=tf.bool)
        batch_size = shape_list(mask)[0]
        mask = tf.concat([tf.constant(np.ones([batch_size, 1]), dtype=tf.bool), mask[:, 1:]], axis=1)

        rnn_outputs = self.lstm(inputs=neigh_vecs, mask=mask)

        from_neighs = self.neigh_weights(rnn_outputs)
        from_self = self.self_weights(self_vecs)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        if self.add_bias:
            output += self.bias

        if self.identity_act: return output
        return self.act(output)


class NodePredict(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.num_classes, kernel_initializer='glorot_uniform', name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(config.agg_dropout)

    def call(self, inputs, training=False):
        node_preds = self.dense(inputs)
        node_preds = self.dropout(node_preds, training=training)
        return node_preds


aggregators = {
    'gcn': GCNAggregator,
    'mean': MeanAggregator,
    'meanpool': MeanPoolingAggregator,
    'maxpool': MaxPoolingAggregator,
    'twomaxpool': TwoMaxLayerPoolingAggregator,
    'seq': SeqAggregator,
    'nodepred': NodePredict
}


def get(aggregator):
    return aggregators.get(aggregator)