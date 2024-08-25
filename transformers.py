"""
Python module with some machine learning Transformer basic blocks based on Tensorflow/Keras

(see "Attention is All You Need", Vaswani et al. 2017)

Example use to make a Generative Pre-Trained Transformer (GPT):
import transformers
import string

NUM_HEADS = 4
HEAD_SIZE = 32

encoder = keras.layers.StringLookup(vocabulary=string.printable)
input_layer = keras.layers.Input(shape=(TIMESERIES_CONTEXT,))
layer = transformers.ByteSplitLayer()(input_layer)
layer = encoder(layer)
layer = transformer.TokenAndPositionEmbedding(encoder.vocabulary_size(), TIMESERIES_CONTEXT, NUM_HEADS*HEAD_SIZE)(layer)
layer = transformer.TransformerDecoder(NUM_HEADS, HEAD_SIZE, dropout=0.5)(layer)
layer = keras.layers.LayerNormalization()(layer)
layer = keras.layers.Dense(encoder.vocabulary_size())(layer)
model = keras.Model(inputs=input_layer, outputs=layer)
model.summary()
"""

import tensorflow as tf
import keras
import numpy as np
from typing import Any
from pydantic.dataclasses import dataclass


class TokenAndPositionEmbedding(keras.layers.Layer):
    """
    Creates an embedding layer that embeds both tokens and their positions.
    For use right before transformer layers that need position embeddings.

    Embeds positions just using incrementing numbers.
    """
    def __init__(self, vocab_size: int, embed_size: int):
        """
        Initialize a TokenAndPositionEmbedding layers

        Requires the embedding output size
        """
        super().__init__()
        self.embed_size = embed_size
        self.C_dimension = vocab_size
        self._positions = None
        self.tok_embedding = None
        self.pos_embedding = None

    def build(self, shape):
        """Build the token and position embedding layers"""
        T_dimension = shape[1]
        self._positions = tf.range(T_dimension)
        self.tok_embedding = keras.layers.Embedding(self.C_dimension, self.embed_size)
        self.pos_embedding = keras.layers.Embedding(T_dimension, self.embed_size)

    def call(self, values, *args, **kwargs):
        """Returns the token plus position embedding values"""
        return self.tok_embedding(values, *args, **kwargs) + self.pos_embedding(self._positions, *args, **kwargs)


class TransformerDecoder(keras.layers.Layer):
    """
    Creates a Transformer Decoder layer

    The Transformer uses multi-head self-attention plus a feed-forward section comprising a nonlinear Dense layer

    It requires inputs that have position embeddings

    The Decoder, uses a causal mask. This means that "future" tokens are masked and the model cannot use them when
    doing its prediction. Use this layer for time-series prediction because it will prevent the model from cheating
    by looking at future tokens.
    """
    def __init__(self, num_heads: int, head_size: int, dropout: float):
        """Initialize a TransformerDecoder layer.

        Requires the number of heads, the head size, and the dropout propotion
        The input layer must have a shape of (batch_size, timeseries_context, num_heads*head_size)
        Dropout must be 0.0 to 1.0

        The output shape will be the same as the input: (batch_size, timeseries_context, num_heads*head_size)
        """
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads, head_size, dropout=dropout)
        self.feed_forward = keras.Sequential([keras.layers.Dense(4*num_heads*head_size, activation='gelu'),
                                              keras.layers.Dense(num_heads*head_size),
                                              keras.layers.Dropout(dropout)])
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()

    def call(self, values, *args, **kwargs):
        """Apply self-attention and a feed-forward layer to the input values"""
        norm_values = self.layer_norm1(values)
        attn = values + self.attention(norm_values, norm_values, use_causal_mask=True, *args, **kwargs)
        norm_attn = self.layer_norm2(attn)
        feed_fwd = attn + self.feed_forward(norm_attn, *args, **kwargs)
        return feed_fwd


class TransformerEncoder(keras.layers.Layer):
    """
    Creates a Transformer Encoder layer

    The Transformer uses multi-head self-attention plus a feed-forward section comprising a nonlinear Dense layer

    It requires inputs that have position embeddings

    The Encoder, unlike the Decoder, does not use a causal mask. Therefore, all of the input tokens can attend
    to any of the other tokens. Don't use this layer for time-series prediction because the model will be able to
    use the current and "future" tokens in its predictions.
    """
    def __init__(self, num_heads: int, head_size: int, dropout: float):
        """Initialize a TransformerDecoder layer.

        Requires the number of heads, the head size, and the dropout propotion
        The input layer must have a shape of (batch_size, timeseries_context, num_heads*head_size)
        Dropout must be 0.0 to 1.0

        The output shape will be the same as the input: (batch_size, timeseries_context, num_heads*head_size)
        """
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads, head_size, dropout=dropout)
        self.feed_forward = keras.Sequential([keras.layers.Dense(4*num_heads*head_size, activation='gelu'),
                                              keras.layers.Dense(num_heads*head_size),
                                              keras.layers.Dropout(dropout)])
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()

    def call(self, values, *args, **kwargs):
        """Apply self-attention and a feed-forward layer to the input values"""
        norm_values = self.layer_norm1(values)
        attn = values + self.attention(norm_values, norm_values, *args, **kwargs)
        norm_attn = self.layer_norm2(attn)
        feed_fwd = attn + self.feed_forward(norm_attn, *args, **kwargs)
        return feed_fwd


class ByteSplitLayer(keras.layers.Layer):
    """
    Splits a 1d input Tensor of strings into a 2d Tensor of individual bytes

    For example:
    layer = ByteSplitLayer()(np.array(['abc', 'def']))
    layer
    <tf.Tensor: shape=(2, 3), dtype=string, numpy=
    array([[b'a', b'b', b'c'],
           [b'd', b'e', b'f']], dtype=object)>
    """
    def __init__(self, **kwargs):
        """Initialize the layer"""
        super().__init__(trainable=False, **kwargs)

    def build(self, shape):
        self.shape = shape

    def call(self, values):
        """Split an input Tensor of strings into a Tensor of individual bytes"""
        if values.shape[0] is None:
            # If we're passed a Tensor without a batch dimension, then return a Tensor representing how many
            # characters we plan to split strings into
            return keras.layers.Flatten()(keras.layers.RepeatVector(self.shape[-1])(values))
        return tf.map_fn(tf.strings.bytes_split, values)


class PerceiverARHeader(keras.layers.Layer):
    """
    Implement a Perceiver Autoregression (AR) head layer
    From the paper: "General-purpose, long-context autoregressive modeling with Perceiver AR"
    https://arxiv.org/abs/2202.07765
    """
    def __init__(self, num_heads: int, head_size: int, output_context: int, dropout: float):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads, head_size, dropout=dropout)
        self.feed_forward = keras.Sequential([keras.layers.Dense(4*num_heads*head_size),
                                              keras.layers.Dense(num_heads*head_size),
                                              keras.layers.Dropout(dropout)])
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()
        self.layer_norm3 = keras.layers.LayerNormalization()
        self._output_context = output_context

    def call(self, values: tf.Tensor, *args, **kwargs):
        query = values[:, -self._output_context:, :]
        norm_query = self.layer_norm1(query, *args, **kwargs)
        norm_value = self.layer_norm2(values, *args, **kwargs)
        attn = query + self.attention(norm_query, norm_value, use_causal_mask=True, *args, **kwargs)
        norm_attn = self.layer_norm3(attn)
        feed_fwd = attn + self.feed_forward(norm_attn)
        return feed_fwd


class PerceiverEncoder(keras.layers.Layer):
    def __init__(self, num_heads: int, head_size: int, dropout: float):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads, head_size, dropout=dropout)
        self.feed_forward = keras.Sequential([keras.layers.Dense(4*num_heads*head_size, activation='gelu'),
                                              keras.layers.Dense(num_heads*head_size),
                                              keras.layers.Dropout(dropout)])
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()
        self.layer_norm3 = keras.layers.LayerNormalization()

    def call(self, query, key):
        norm_query = self.layer_norm1(query)
        norm_key = self.layer_norm2(key)
        attn = query + self.attention(norm_query, norm_key)
        norm_attn = self.layer_norm3(attn)
        feed_fwd = attn + self.feed_forward(norm_attn)
        return feed_fwd


class PerceiverProcessor(TransformerEncoder):
    """
    Perceiver processor layer (processes the latent tensors in the inner layers of the Perceiver)
    It's the same as a Transformer self-attention layer with no causal mask.
    """
    pass


class PerceiverDecoder(keras.layers.Layer):
    def __init__(self, num_heads, head_size, output_dimension, dropout):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads, head_size, output_shape=(output_dimension,), dropout=dropout)
        self.feed_forward = keras.Sequential([keras.layers.Dense(4*output_dimension, activation='gelu'),
                                              keras.layers.Dense(output_dimension),
                                              keras.layers.Dropout(dropout)])
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()

    def call(self, query, values):
        # Don't normalize the query or create a residual for it (per the Perceiver paper)
        norm_values = self.layer_norm1(values)
        attn = self.attention(query, norm_values)
        norm_attn = self.layer_norm2(attn)
        feed_fwd = attn + self.feed_forward(norm_attn)
        return feed_fwd


class PadAndTruncate(keras.layers.Layer):
    """
    Pads or truncates the second dimension of a Tensor to shape it to a consistent size

    For example:
    ```
    pt = PadAndTruncate()
    pt(keras.layers.Input(shape=(3,)))
    pt(tf.constant([[1], [2], [3]]))
    -> <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
       array([[1, 0, 0],
              [2, 0, 0],
              [3, 0, 0]], dtype=int32)>
    
    pt(tf.constant([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]))
    -> <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
       array([[3, 4, 5],
              [4, 5, 6],
              [5, 6, 7]], dtype=int32)>
    ```
    """
    def __init__(self, size: int=None, pad_with: Any=None) -> None:
        super().__init__()
        self._input_shape = size
        self._pad_with = pad_with
        if self._pad_with is None:
            self._pad_with = 0
    
    def build(self, input_shape):
        if self._input_shape is None:
            self._input_shape = input_shape
    
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        if inputs.shape[1] < self._input_shape[1]:
            pad_length = self._input_shape[1] - inputs.shape[1]
            paddings = [[0, 0], [0, pad_length]] + [0, 0] * (len(self._input_shape) - 2)
            padded_tensor = tf.pad(inputs, paddings, "CONSTANT", self._pad_with)
            return padded_tensor
        else:
            truncated_tensor = inputs[:, -self._input_shape[1]:]
            return truncated_tensor


def generate_batches(data: np.array, context_length: int, batch_size: int, shuffle=True):
    """
    Generate time-slice batches of data of a given context length from 1-dimensional source data
    The batches are tuples of values and targets: values are used as input to the prediction model;
    targets are used as the outputs to predict. Targets are ahead of the values in time by one.
    """
    # List all of the possible starting offsets for time-slice windows in the data
    starting_offsets = np.arange(len(data)-context_length)
    if shuffle:
        np.random.shuffle(starting_offsets)
    for batch_idx in range(0, len(starting_offsets), batch_size):
        # Choose some starting offsets to make up this batch
        # Broadcast the starting offsets into a 2-D array by using Numpy broadcasting
        # This will create a 2-D array of shape (batch_size, context_size) that can be used to index into the data
        batch_indices = starting_offsets[batch_idx:(batch_idx+batch_size), np.newaxis] + np.arange(context_length)
        values = data[batch_indices]
        targets = data[batch_indices+1]
        yield values, targets


class InverseTokenEmbedding(keras.layers.Layer):
    """Inverse token embedding"""
    def __init__(self, tok_embedding: keras.layers.Embedding) -> None:
        super().__init__()
        self._tok_embedding = tok_embedding

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(inputs, self._tok_embedding.weights[0], transpose_b=True)


@dataclass
class GPTConfig:
    vocab_size: int
    context: int
    head_size: int
    num_heads: int
    num_layers: int
    dropout: float


class GPT(keras.layers.Layer):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self._tok_embedding = keras.layers.Embedding(config.vocab_size, config.num_heads*config.head_size)
        self._pos_embedding = keras.layers.Embedding(config.context, config.num_heads*config.head_size)
        self._transformers = keras.Sequential([TransformerDecoder(config.num_heads,
                                                                  config.head_size,
                                                                  dropout=config.dropout)]*config.num_layers)
        self._layer_norm = keras.layers.LayerNormalization()
        self._inv_tok_embedding = InverseTokenEmbedding(self._tok_embedding)
        self._positions = tf.range(config.context)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        layer = self._tok_embedding(inputs) + self._pos_embedding(self._positions)
        layer = self._transformers(layer)
        layer = self._layer_norm(layer)
        layer = self._inv_tok_embedding(layer)
        return layer
