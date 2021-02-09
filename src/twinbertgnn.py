"""TwinBert Implementation"""

# import torch
import logging
from tensorflow.python.keras import backend as K
from transformers.modeling_tf_bert import TFSequenceClassificationLoss, TFBertPreTrainedModel
from transformers import BertConfig
import tensorflow as tf
from tensorflow.python.keras.saving.hdf5_format import load_attributes_from_hdf5_group
from transformers.modeling_tf_utils import hf_bucket_url
from transformers.file_utils import TF2_WEIGHTS_NAME, cached_path
import h5py
import numpy as np
from transformers.modeling_tf_utils import shape_list
import os

try:
    from bert_core import *
    from encoders import *
except ImportError:
    from .bert_core import *
    from .encoders import *


logger = logging.getLogger(__name__)


# TwinBert pooler layer, if 'clspooler', simply use the vector corresponding to the pooled bert output. Otherwise use an attention weighting to weight the token vectors
class PoolerLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(PoolerLayer, self).__init__(**kwargs)
        self.config = config
        if self.config.pooler_type == 'weightpooler':
            self.weighting = tf.keras.layers.Dense(1, name='weighted_pooler')

    def call(self, cls_tensor, term_tensor, mask, training=False):
        if self.config.pooler_type == 'clspooler':
            return cls_tensor
        elif self.config.pooler_type == 'weightpooler':
            weights = self.weighting(term_tensor)
            weights = weights + tf.expand_dims((tf.cast(mask, weights.dtype) - 1.0), axis=2) / 1e-8
            weights = tf.nn.softmax(weights, axis=1)
            return tf.reduce_sum(tf.multiply(term_tensor, weights), axis=1)
        elif self.config.pooler_type == 'average':
            inds = tf.cast(mask, tf.float32)
            output = tf.transpose(term_tensor, [2, 0, 1]) * inds
            token_tensor = tf.reduce_sum(tf.transpose(output, [1, 2, 0]), axis=1)
            token_tensor = tf.transpose(token_tensor, [1, 0]) / tf.reduce_sum(inds, axis=-1)
            return tf.transpose(token_tensor, [1, 0])


# TwinBert postprocessing layer (downscale, tanh pooling, and quantization)
class PostprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(PostprocessingLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, downscale, vec, is_left=True, training=False):
        def quantization(v):
            v = tf.round((v + 1) / (2 / 255))  # 2/256, it is 2/255 in production
            v = v * (2.0 / 255) - 1
            return v

        if self.config.downscale > 0:
            vec = downscale(vec)
        if self.config.tanh_pooler:
            vec = tf.tanh(vec)
        if self.config.quantization_side == 'both' or (
                (not is_left) and self.config.quantization_side == 'right') or (
                is_left and self.config.quantization_side == 'left'):
            vec = quantization(vec)
        return vec


# TwinBert Crossing layer to combine Q and K vectors into a score
class CrossingLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(CrossingLayer, self).__init__(**kwargs)
        self.config = config

        if self.config.sim_type == 'cosine':
            self.logistic = tf.keras.layers.Dense(1, name='logistic')
        elif self.config.sim_type == 'feedforward':
            self.ff_dense0 = tf.keras.layers.Dense(self.config.res_size, name='ff_dense0')

            dim_scale_gnn = 2 if (self.config.agg_concat and self.config.gnn_model == "graphsage") else 1
            gnn_concat_ret_scale = 1 if self.config.gnn_concat_residual else 0

            input_a_dim = self.config.hidden_size * gnn_concat_ret_scale + self.config.hidden_dims[-1] * dim_scale_gnn
            input_b_dim = self.config.hidden_size * gnn_concat_ret_scale + self.config.hidden_dims[-1] * dim_scale_gnn
            if self.config.a_fanouts[0] == 0:
                input_a_dim = self.config.hidden_size
            if self.config.b_fanouts[0] == 0:
                input_b_dim = self.config.hidden_size

            self.ff_dense1 = tf.keras.layers.Dense((input_a_dim if self.config.comb_type == 'max' else (input_a_dim + input_b_dim)),
                                                   name='ff_dense1')

            if self.config.res_bn:
                self.res_bn_1 = tf.keras.layers.BatchNormalization(axis=-1, name='batch_norm_1')
                self.res_bn_2 = tf.keras.layers.BatchNormalization(axis=-1, name='batch_norm_2')

            self.relu = tf.keras.layers.ReLU()
            self.logistic = tf.keras.layers.Dense(2, name='logistic')

    def call(self, vec_a, vec_b, training=False):
        if self.config.sim_type == 'cosine':
            sim_score = tf.reduce_sum(
                tf.multiply(tf.math.l2_normalize(vec_a, axis=1), tf.math.l2_normalize(vec_b, axis=1)), axis=1,
                keepdims=True)
            probabilities = tf.math.sigmoid(self.logistic(sim_score))
            probabilities = tf.stack([1 - probabilities, probabilities], axis=-1)

        elif self.config.sim_type == 'feedforward':
            if self.config.comb_type == 'max':
                cross_input = tf.math.maximum(vec_a, vec_b)
            elif self.config.comb_type == 'concat':
                cross_input = tf.concat([vec_a, vec_b], axis=1)

            output = self.ff_dense0(cross_input)
            if self.config.res_bn:
                output = self.res_bn_1(output, training=training)

            output = self.ff_dense1(self.relu(output))

            if self.config.res_bn:
                output = self.res_bn_2(output, training=training)

            if self.config.crossing_res:
                output = self.relu(output + cross_input)
            logits = self.logistic(output)
            probabilities = tf.nn.softmax(logits, axis=-1)

        return probabilities


class TwinBERTGNNCore(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TwinBERTGNNCore, self).__init__(**kwargs)
        self.config = config

        self.bert_encoder_a = BERTCore(self.config, trainable=self.config.bert_trainable)
        self.pooler_a = PoolerLayer(self.config, name='pooler_a', trainable=self.config.bert_trainable)
        if self.config.post_processing:
            self.postprocessing = PostprocessingLayer(self.config, name='postprocessing')
        self.crossing = CrossingLayer(self.config, name='crossing')
        if self.config.use_two_crossings:
            self.tb_crossing = CrossingLayer(self.config, name='twinbert_crossing')
        else:
            self.tb_crossing = self.crossing

        if self.config.downscale > 0:
            self.downscale_a = tf.keras.layers.Dense(self.config.downscale, name='downscale_a',
                                                     trainable=self.config.bert_trainable)
        else:
            self.downscale_a = None

        encoder_class = encoders.get(self.config.gnn_model.lower())
        self.encoder_a = encoder_class(self.config, self.config.a_fanouts)
        if self.config.use_two_gnn:
            self.encoder_b = encoder_class(self.config, self.config.b_fanouts)
        else:
            self.encoder_b = self.encoder_a

        if self.config.use_two_bert:
            self.bert_encoder_b = BERTCore(self.config, trainable=self.config.bert_trainable)
            self.pooler_b = PoolerLayer(self.config, name='pooler_b', trainable=self.config.bert_trainable)

            if self.config.downscale > 0:
                self.downscale_b = tf.keras.layers.Dense(self.config.downscale, name='downscale_b',
                                                         trainable=self.config.bert_trainable)
            else:
                self.downscale_b = None
        else:
            self.bert_encoder_b = self.bert_encoder_a
            self.pooler_b = self.pooler_a
            if self.config.downscale > 0:
                self.downscale_b = self.downscale_a
            else:
                self.downscale_b = None


    def call(self, inputs, training=False):
        input_ids_as = {"input_ids_a_0": tf.reshape(inputs.get("input_ids_a_0", None), [-1, self.config.max_seq_len])}
        attention_mask_as = {"attention_mask_a_0": tf.reshape(inputs.get("attention_mask_a_0", None), [-1, self.config.max_seq_len])}

        if len(self.config.a_fanouts) > 0 and self.config.a_fanouts[0] > 0:
            for i in range(1, len(self.config.a_fanouts) + 1):
                input_ids_as["input_ids_a_" + str(i)] = tf.reshape(inputs.get("input_ids_a_" + str(i), None),
                                                                   [-1, self.config.max_seq_len])
                attention_mask_as["attention_mask_a_" + str(i)] = tf.reshape(
                    inputs.get("attention_mask_a_" + str(i), None), [-1, self.config.max_seq_len])

        input_ids_bs = {"input_ids_b_0": tf.reshape(inputs.get("input_ids_b_0", None), [-1, self.config.max_seq_len])}
        attention_mask_bs = {"attention_mask_b_0": tf.reshape(inputs.get("attention_mask_b_0", None), [-1, self.config.max_seq_len])}

        if len(self.config.b_fanouts) > 0 and self.config.b_fanouts[0] > 0:
            for i in range(1, len(self.config.b_fanouts) + 1):
                input_ids_bs["input_ids_b_" + str(i)] = tf.reshape(inputs.get("input_ids_b_" + str(i), None),
                                                                   [-1, self.config.max_seq_len])
                attention_mask_bs["attention_mask_b_" + str(i)] = tf.reshape(
                    inputs.get("attention_mask_b_" + str(i), None), [-1, self.config.max_seq_len])

        berts_a = {}
        berts_b = {}

        # q d q d...
        for i in range(len(input_ids_as)):
            if i % 2 == 0:
                term_tensor, cls_tensor = self.bert_encoder_a(input_ids_as["input_ids_a_" + str(i)],
                                                              attention_mask=attention_mask_as[
                                                                  "attention_mask_a_" + str(i)],
                                                              output_attentions=False, output_hidden_states=False,
                                                              training=training)
                vec = self.pooler_a(cls_tensor, term_tensor, attention_mask_as["attention_mask_a_" + str(i)],
                                    training=training)
                if self.config.post_processing:
                    vec = self.postprocessing(self.downscale_a, vec, is_left=True, training=training)
            else:
                term_tensor, cls_tensor = self.bert_encoder_a(input_ids_as["input_ids_a_" + str(i)],
                                                              attention_mask=attention_mask_as[
                                                                  "attention_mask_a_" + str(i)],
                                                              output_attentions=False, output_hidden_states=False,
                                                              training=training)
                vec = self.pooler_b(cls_tensor, term_tensor, attention_mask_as["attention_mask_a_" + str(i)],
                                    training=training)
                if self.config.post_processing:
                    vec = self.postprocessing(self.downscale_b, vec, is_left=False, training=training)
            berts_a["bert_" + str(i)] = vec

        # d q d q...
        for i in range(len(input_ids_bs)):
            if i % 2 == 0:
                term_tensor, cls_tensor = self.bert_encoder_b(input_ids_bs["input_ids_b_" + str(i)],
                                                              attention_mask=attention_mask_bs[
                                                                  "attention_mask_b_" + str(i)],
                                                              output_attentions=False, output_hidden_states=False,
                                                              training=training)
                vec = self.pooler_b(cls_tensor, term_tensor, attention_mask_bs["attention_mask_b_" + str(i)],
                                    training=training)
                if self.config.post_processing:
                    vec = self.postprocessing(self.downscale_b, vec, is_left=False, training=training)
            else:
                term_tensor, cls_tensor = self.bert_encoder_a(input_ids_bs["input_ids_b_" + str(i)],
                                                              attention_mask=attention_mask_bs[
                                                                  "attention_mask_b_" + str(i)],
                                                              output_attentions=False, output_hidden_states=False,
                                                              training=training)
                vec = self.pooler_a(cls_tensor, term_tensor, attention_mask_bs["attention_mask_b_" + str(i)],
                                    training=training)
                if self.config.post_processing:
                    vec = self.postprocessing(self.downscale_a, vec, is_left=True, training=training)
            berts_b["bert_" + str(i)] = vec

        if self.config.gnn_model == 'weighted':
            for i in range(self.config.head_nums[0]):
                if self.config.weighted_gnn_type[i] == 'ctr':
                    berts_a["weights_" + str(i)] = tf.cast(inputs.get('click_a_1', None), dtype=tf.float32) / tf.cast(inputs.get('impression_a_1', None), dtype=tf.float32)
                    berts_b["weights_" + str(i)] = tf.cast(inputs.get('click_b_1', None), dtype=tf.float32) / tf.cast(inputs.get('impression_b_1', None), dtype=tf.float32)
                else:
                    berts_a["weights_" + str(i)] = tf.math.maximum(tf.math.log(tf.cast(inputs.get(self.config.weighted_gnn_type[i] + '_a_1', None), dtype=tf.float32) + 1.0 + 1e-7), 0.0)
                    berts_b["weights_" + str(i)] = tf.math.maximum(tf.math.log(tf.cast(inputs.get(self.config.weighted_gnn_type[i] + '_b_1', None), dtype=tf.float32) + 1.0 + 1e-7), 0.0)

        # berts_a_input = tf.identity(berts_a["bert_0"])
        # berts_b_input = tf.identity(berts_b["bert_0"])
        # # print(berts_a)
        # print("bert_0_a_input")
        # print(berts_a_input)
        vec_a = self.encoder_a(berts_a)
        # print("vec_a")
        # print(vec_a)
        vec_b = self.encoder_b(berts_b)
        if self.config.gnn_concat_residual:
            vec_a = tf.concat([vec_a, berts_a["bert_0"]], axis=1)
            vec_b = tf.concat([vec_b, berts_b["bert_0"]], axis=1)
        elif self.config.gnn_add_residual:
            vec_a = vec_a + berts_a["bert_0"]
            vec_b = vec_b + berts_b["bert_0"]

        # print("bert_0_a")
        # print(berts_a["bert_0"])
        # print("vec_a_res")
        # print(vec_a)
        # print(berts_a_input - berts_a["bert_0"])
        output = self.crossing(vec_a, vec_b, training=training)
        tb_output = None
        if self.config.tb_loss:
            tb_output = self.tb_crossing(berts_a["bert_0"], berts_b["bert_0"], training=training)
        return output, tb_output, vec_a, vec_b


class TwinBERTGNN(TFBertPreTrainedModel):
    def __init__(self, config_file, **kwargs):
        self.config = self.init_config_from_file(config_file)
        super(TwinBERTGNN, self).__init__(self.config, **kwargs)
        self.twinbertgnncore = TwinBERTGNNCore(self.config, name="twin_bert")

    @property
    def dummy_inputs(self):
        length = self.config.max_n_letters * self.config.max_seq_len if self.config.embedding_type == "triletter" else self.config.max_seq_len

        input_dict = {
            "input_ids_a_0": tf.ones([1, length], dtype=tf.int32),
            "attention_mask_a_0": tf.ones([1, self.config.max_seq_len], dtype=tf.int32),
            "inputs_embeds_a_0": None,
            "input_ids_b_0": tf.ones([1, length], dtype=tf.int32),
            "attention_mask_b_0": tf.ones([1, self.config.max_seq_len], dtype=tf.int32),
            "inputs_embeds_b_0": None,
        }

        if len(self.config.a_fanouts) > 0 and self.config.a_fanouts[0] > 0:
            layer_node = 1
            for layer in range(len(self.config.a_fanouts)):
                layer_node *= self.config.a_fanouts[layer]
                input_dict.update({
                    'input_ids_a_' + str(layer + 1): tf.ones([1, layer_node, length], dtype=tf.int32),
                    "attention_mask_a_" + str(layer + 1): tf.ones([1, layer_node, self.config.max_seq_len],
                                                                  dtype=tf.int32),
                    'impression_a_' + str(layer + 1): tf.ones([1, layer_node], dtype=tf.int32),
                    'click_a_' + str(layer + 1): tf.ones([1, layer_node], dtype=tf.int32)
                })

        if len(self.config.b_fanouts) > 0 and self.config.b_fanouts[0] > 0:
            layer_node = 1
            for layer in range(len(self.config.b_fanouts)):
                layer_node *= self.config.b_fanouts[layer]
                input_dict.update({
                    'input_ids_b_' + str(layer + 1): tf.ones([1, layer_node, length], dtype=tf.int32),
                    "attention_mask_b_" + str(layer + 1): tf.ones([1, layer_node, self.config.max_seq_len],
                                                                  dtype=tf.int32),
                    'impression_b_' + str(layer + 1): tf.ones([1, layer_node], dtype=tf.int32),
                    'click_b_' + str(layer + 1): tf.ones([1, layer_node], dtype=tf.int32)
                })

        return input_dict

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def compute_loss(self, labels, outputs, loss_type):
        probabilities = outputs

        if loss_type == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            pred = probabilities[:, 1][:, tf.newaxis]
            labels = tf.cast(labels, tf.float32)
        elif loss_type == "ssce":
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )
            float_labels = tf.cast(labels, tf.float32)
            model_label = tf.stack([1 - float_labels, float_labels], axis=-1)
            labels = tf.nn.softmax(model_label, -1)
            pred = probabilities
        else:
            if loss_type != "ce":
                logger.info('unknown loss type {}; fallback to ce'.format(loss_type))
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )
            pred = probabilities

        return loss_fn(labels, pred)

    def call(self, inputs, labels=None, training=False, **kwargs):
        outputs = self.twinbertgnncore(inputs, training=training, **kwargs)

        if labels is not None:
            loss = self.compute_loss(labels, outputs[0], self.config.loss_type)
            if self.config.tb_loss:
                loss += self.compute_loss(labels, outputs[1], self.config.loss_type)
            outputs = (loss,) + outputs

        return outputs

    @classmethod
    def init_config_from_file(cls, filename):
        ret = {}
        with open(filename, 'r', encoding='utf-8') as fp:
            while True:
                line = fp.readline().strip('\n\r')
                if line == '':
                    break
                tokens = line.split('\t')
                name = tokens[0].split(':')[0]
                type = tokens[0].split(':')[1]
                val = tokens[1]

                if type == 'str':
                    ret[name] = val
                elif type == 'int':
                    ret[name] = int(val)
                elif type == 'float':
                    ret[name] = float(val)
                elif type == 'bool':
                    ret[name] = (val == 'True')
                else:
                    print('unrecognized config: ' + line)
        ret = BertConfig.from_dict(ret)
        ret.a_fanouts = list(map(int, ret.a_fanouts.split(","))) if ret.a_fanouts else []
        ret.b_fanouts = list(map(int, ret.b_fanouts.split(","))) if ret.b_fanouts else []
        ret.hidden_dims = list(map(int, ret.hidden_dims.split(","))) if ret.hidden_dims else []
        ret.gnn_acts = ret.gnn_acts.split(",") if ret.gnn_acts else []
        ret.head_nums = list(map(int, ret.head_nums.split(","))) if ret.head_nums else []
        ret.weighted_gnn_type = ret.weighted_gnn_type.split(",") if ret.weighted_gnn_type else []
        return ret

    @classmethod
    def load_from_checkpoint(cls, config_file, checkpoint_file, checkpoint_dict_file, is_tf_checkpoint=True, **kwargs):
        def _read_checkpoint_dict(filename):
            ret = {}
            with open(filename, 'r', encoding='utf-8') as fp:
                while True:
                    line = fp.readline().strip('\n\r')
                    if line == '':
                        break
                    tokens = line.split('\t')
                    model_weights_name = tokens[0]
                    ckpt_weights_name = tokens[1]

                    ret[model_weights_name] = ckpt_weights_name
            return ret

        model = cls(config_file, **kwargs)
        model(model.dummy_inputs, training=False)
        checkpoint_dict = _read_checkpoint_dict(checkpoint_dict_file)
        # print(checkpoint_dict)

        weight_value_tuples = []
        w_names = []
        if is_tf_checkpoint:
            tf_checkpoint_reader = tf.train.load_checkpoint(checkpoint_file)
            for w in model.layers[0].weights:
                w_name = '/'.join(w.name.split('/')[3:])

                if w_name in checkpoint_dict:
                    weight_value_tuples.append((w, tf_checkpoint_reader.get_tensor(checkpoint_dict[w_name])))
                    w_names.append(w_name)
                else:
                    print(w_name)
        else:
            torch_checkpoint = torch.load(checkpoint_file)

            for w in model.layers[0].weights:
                if w.name not in checkpoint_dict:
                    continue
                if w.name.split('/')[-1] == "kernel:0":
                    weight_value_tuples.append((w, torch_checkpoint[checkpoint_dict[w.name]].transpose(0, 1).numpy()))
                else:
                    weight_value_tuples.append((w, torch_checkpoint[checkpoint_dict[w.name]].numpy()))
                w_names.append(w.name)

        K.batch_set_value(weight_value_tuples)

        print("Loaded %d weights" % (len(w_names)))
        print("Loaded weights names are: %s" % (", ".join(w_names)))

        model(model.dummy_inputs, training=False)
        return model

    @classmethod
    def load_from_bert_pretrained(cls, config_file, pretrained_model_name='bert-base-uncased', **kwargs):
        model = cls(config_file, **kwargs)
        model(model.dummy_inputs, training=False)

        ckpt_layer_mapping = {}
        for vind, ckpt_ind in enumerate(model.config.ckpt_layer_mapping.split(',')):
            ckpt_layer_mapping['layer_._{}'.format(vind)] = 'layer_._{}'.format(ckpt_ind)

        archive_file = hf_bucket_url(pretrained_model_name, filename=TF2_WEIGHTS_NAME, use_cdn=True)
        resolved_archive_file = cached_path(archive_file, cache_dir=None, force_download=False, resume_download=False,
                                            proxies=None)
        f = h5py.File(resolved_archive_file, mode='r')

        layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
        g = f[layer_names[0]]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        weights_map = {'/'.join(name.split('/')[2:]): i for i, name in enumerate(weight_names)}
        weight_value_tuples = []
        w_names = []
        for w in model.layers[0].weights:
            w_name = '/'.join(w.name.split('/')[3:])
            for k in ckpt_layer_mapping:
                if w_name.find(k):
                    w_name = w_name.replace(k, ckpt_layer_mapping[k])
                    break

            if w_name in weights_map and w.shape == weight_values[weights_map[w_name]].shape:
                w_names.append(w_name)
                weight_value_tuples.append((w, weight_values[weights_map[w_name]]))

        logger.info("Loaded %d weights" % (len(w_names)))
        logger.info("Loaded weights names are: %s" % (", ".join(w_names)))

        K.batch_set_value(weight_value_tuples)

        print("Loaded %d weights" % (len(w_names)))
        print("Loaded weights names are: %s" % (", ".join(w_names)))

        model(model.dummy_inputs, training=False)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path, config_path, **kwargs):
        model = cls(config_path, **kwargs)
        model(model.dummy_inputs, training=False)  # build the network with dummy inputs

        assert os.path.isfile(pretrained_model_path), "Error retrieving file {}".format(pretrained_model_path)
        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            model.load_weights(pretrained_model_path, by_name=True)
        except OSError:
            raise OSError(
                "Unable to load weights from h5 file. "
                "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
            )

        model(model.dummy_inputs, training=False)  # Make sure restore ops are run
        return model