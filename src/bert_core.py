"""Generalize BERT model with Triletter or Simpiied BPE encoder"""

import tensorflow as tf
from transformers import BertConfig, TFBertMainLayer, TFBertPreTrainedModel
from transformers.modeling_tf_bert import TFBertEncoder, TFBertPooler, TFBertEmbeddings, TFBertPredictionHeadTransform
from transformers.modeling_tf_utils import keras_serializable, shape_list, get_initializer
from transformers.tokenization_utils import BatchEncoding


class TFBertEmbeddingsSimple(tf.keras.layers.Layer):
    """Construct the embeddings with only word embeddings.
    """

    def __init__(self, config, **kwargs):
        super(TFBertEmbeddingsSimple, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=get_initializer(self.initializer_range), name='position_embeddings')

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape):
        with tf.name_scope("word_embeddings"):
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range)
            )

    def call(self, inputs, training=False):
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs
        input_shape = shape_list(input_ids)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class TriletterEmbeddings(tf.keras.layers.Layer):
    """Comparing with TriletterEmbeddingsSimple, this one has position encoding, name is a little bit ugly, but not breaking anything
    """

    def __init__(self, config, **kwargs):
        super(TriletterEmbeddings, self).__init__(**kwargs)
        self.triletter_max_letters_in_word = config.triletter_max_letters_in_word  # 20, so 20 triletters
        self.triletter_embeddings = tf.keras.layers.Embedding(config.vocab_size + 1, config.hidden_size,
                                                              mask_zero=True, name='triletter_embeddings')
        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings + 1,
                                                             config.hidden_size, mask_zero=True,
                                                             name='position_embeddings')

    def call(self, inputs, training=False):
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs
        triletter_max_seq_len = shape_list(input_ids)[1] // self.triletter_max_letters_in_word
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.triletter_embeddings(input_ids)  # [N, 12*[20], hidden_size]

        embeddings = tf.reshape(embeddings, [-1, triletter_max_seq_len, self.triletter_max_letters_in_word,
                                             shape_list(embeddings)[-1]])
        embeddings = tf.reshape(tf.reduce_sum(embeddings, axis=2),
                                [-1, triletter_max_seq_len, shape_list(embeddings)[-1]])

        embeddings = embeddings + position_embeddings

        return embeddings


class BERTCore(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config, **kwargs):
        super(BERTCore, self).__init__(**kwargs)
        self.config = config

        self.num_hidden_layers = self.config.num_hidden_layers
        self.initializer_range = self.config.initializer_range
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

        if self.config.embedding_type == 'triletter':
            self.embeddings = TriletterEmbeddings(self.config)
        elif self.config.embedding_type == 'bpe_simple':
            self.embeddings = TFBertEmbeddingsSimple(self.config, name="embeddings")
        else:
            self.embeddings = TFBertEmbeddings(self.config, name="embeddings")

        self.encoder = TFBertEncoder(self.config, name="encoder")
        self.pooler = TFBertPooler(self.config, name='pooler')

    def get_input_embeddings(self):
        return self.embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(
            self,
            inputs,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            output_attentions = inputs[6] if len(inputs) > 6 else output_attentions
            output_hidden_states = inputs[7] if len(inputs) > 7 else output_hidden_states
            assert len(inputs) <= 8, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 8, "Too many inputs."
        else:
            input_ids = inputs

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            if type(self.embeddings) == TriletterEmbeddings:
                attention_mask = tf.ones(
                    [input_shape[0], input_shape[1] // self.embeddings.triletter_max_letters_in_word])
            else:
                attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            if type(self.embeddings) == TriletterEmbeddings:
                token_type_ids = tf.zeros(
                    [input_shape[0], input_shape[1] // self.embeddings.triletter_max_letters_in_word])
            else:
                token_type_ids = tf.fill(input_shape, 0)
        if position_ids is None:
            if type(self.embeddings) == TriletterEmbeddings:
                position_ids = (tf.range(input_shape[1] // self.embeddings.triletter_max_letters_in_word,
                                         dtype=tf.int32) + 1)[tf.newaxis, :]
            else:
                position_ids = tf.range(int(input_shape[1]), dtype=tf.int32)[tf.newaxis, :]
            position_ids = tf.where(attention_mask == 0, tf.zeros_like(position_ids), position_ids)

        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, head_mask, output_attentions, output_hidden_states],
            training=training,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BERTModel(TFBertPreTrainedModel):
    def prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def __init__(self, config):
        super(BERTModel, self).__init__(config)
        self.config = config
        self.bert = BERTCore(self.config, name="bert")

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        return outputs
