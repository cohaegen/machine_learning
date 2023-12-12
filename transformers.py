"""
Python module with some machine learning Transformer basic blocks based on Tensorflow/Keras

(see "Attention is All You Need", Vaswani et al. 2017)

Example use to make a Generative Pre-Trained Transformer (GPT):
import transformers
import string

NUM_HEADS = 4
HEAD_SIZE = 32

encoder = tf.keras.layers.StringLookup(vocabulary=string.printable)
input_layer = tf.keras.layers.Input(shape=(TIMESERIES_CONTEXT,))
layer = transformers.ByteSplitLayer()(input_layer)
layer = encoder(layer)
layer = transformer.TokenAndPositionEmbedding(encoder.vocabulary_size(), TIMESERIES_CONTEXT, NUM_HEADS*HEAD_SIZE)(layer)
layer = transformer.TransformerDecoder(NUM_HEADS, HEAD_SIZE, dropout=0.5)(layer)
layer = tf.keras.layers.LayerNormalization()(layer)
layer = tf.keras.layers.Dense(encoder.vocabulary_size())(layer)
model = tf.keras.Model(inputs=input_layer, outputs=layer)
model.summary()
"""

import tensorflow as tf
import numpy as np

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """
    Creates an embedding layer that embeds both tokens and their positions.
    For use right before transformer layers that need position embeddings.

    Embeds positions just using incrementing numbers.
    """
    def __init__(self, vocab_size: int, context_size: int, embed_size: int):
        """
        Initialize a TokenAndPositionEmbedding layers

        Requires the token vocabulary size, the length of inputs (to create the position embedding),
        and the embedding output size
        """
        super().__init__()
        self.tok_embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.pos_embedding = tf.keras.layers.Embedding(context_size, embed_size)
        self._positions = tf.range(context_size)
    
    def call(self, values):
        """Returns the token plus position embedding values"""
        return self.tok_embedding(values) + self.pos_embedding(self._positions)


class TransformerDecoder(tf.keras.layers.Layer):
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
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, head_size, dropout=dropout)
        self.feed_forward = tf.keras.Sequential([tf.keras.layers.Dense(4*num_heads*head_size, activation='gelu'),
                                                 tf.keras.layers.Dense(num_heads*head_size),
                                                 tf.keras.layers.Dropout(dropout)])
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, values, *args, **kwargs):
        """Apply self-attention and a feed-forward layer to the input values"""
        norm_values = self.layer_norm1(values)
        attn = values + self.attention(norm_values, norm_values, use_causal_mask=True, *args, **kwargs)
        norm_attn = self.layer_norm2(attn)
        feed_fwd = attn + self.feed_forward(norm_attn, *args, **kwargs)
        return feed_fwd


class TransformerEncoder(tf.keras.layers.Layer):
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
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, head_size, dropout=dropout)
        self.feed_forward = tf.keras.Sequential([tf.keras.layers.Dense(4*num_heads*head_size, activation='gelu'),
                                                 tf.keras.layers.Dense(num_heads*head_size),
                                                 tf.keras.layers.Dropout(dropout)])
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, values, *args, **kwargs):
        """Apply self-attention and a feed-forward layer to the input values"""
        norm_values = self.layer_norm1(values)
        attn = values + self.attention(norm_values, norm_values, *args, **kwargs)
        norm_attn = self.layer_norm2(attn)
        feed_fwd = attn + self.feed_forward(norm_attn, *args, **kwargs)
        return feed_fwd


class ByteSplitLayer(tf.keras.layers.Layer):
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
            return tf.keras.layers.Flatten()(tf.keras.layers.RepeatVector(self.shape[-1])(values))
        return tf.map_fn(tf.strings.bytes_split, values)