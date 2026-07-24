# import
import pandas as pd
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import os
from pathlib import Path


# ---------------------------------------------------------------------------- #
#                              positional encoding                             #
# ---------------------------------------------------------------------------- #

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis] # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth # (1, depth)

    angle_rates = 1 / (10_000**depths) # (1, depth)
    angle_rads = positions * angle_rates # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    return tf.cast(pos_encoding, dtype=tf.float32)

# ---------------------------------------------------------------------------- #

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,  d_model, max_length=2048):
        super().__init__()
        self.d_model = d_model
        self.projection = tf.keras.layers.Dense( d_model)
        self.pos_encoding = positional_encoding(length=max_length, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]

        # project features into model space
        x = self.projection(x)

        # scale
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # add in positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x
    

# ---------------------------------------------------------------------------- #
#                             Base Attention Layer                             #
# ---------------------------------------------------------------------------- #

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


# ---------------------------------------------------------------------------- #
#                             Cross Attention Layer                            #
# ---------------------------------------------------------------------------- #


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # cache the attention scores for plotting
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x
    

# ---------------------------------------------------------------------------- #
#                             Global Self Attention                            #
# ---------------------------------------------------------------------------- #

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x
    

# ---------------------------------------------------------------------------- #
#                          Causal Self-Attention Layer                         #
# ---------------------------------------------------------------------------- #

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True # only difference from the global self-attn layer
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

# ---------------------------------------------------------------------------- #
#                             Feed Forward Network                             #
# ---------------------------------------------------------------------------- #


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    

# ---------------------------------------------------------------------------- #
#                                    Encoder                                   #
# ---------------------------------------------------------------------------- #

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
# ---------------------------------------------------------------------------- #

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, #vocab_size,
                 dropout_rate=0.1
                 ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model)

        self.enc_layer = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate
                         )
            for _ in range(num_layers)
        ]
        self.drop_out = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # x is the token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)

        # dropout
        x = self.drop_out(x)

        for i in range(self.num_layers):
            x = self.enc_layer[i](x)

        return x # Shape (batch_size, seq_len, d_model)
    

# ---------------------------------------------------------------------------- #
#                                    decoder                                   #
# ---------------------------------------------------------------------------- #

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1
                 ):

        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x) # shape (batch_size, seq_len, d_model)
        return x
    
# ---------------------------------------------------------------------------- #

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # 'x'
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x
    

# ---------------------------------------------------------------------------- #
#                                the Transformer                               #
# ---------------------------------------------------------------------------- #

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               dropout_rate=dropout_rate
                               )
        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               dropout_rate=dropout_rate
                               )

        self.final_layer = tf.keras.layers.Dense(1) # 

    def call(self, inputs):
            # If inputs is a single tensor, handle it directly
        if isinstance(inputs, (list, tuple)):
            context, x = inputs
        else:
            # Split the input tensor if it's combined
            # Or use the same input for both context and target
            context = inputs
            x = inputs

        context = self.encoder(context) # (batch_sie, context_len, d_model)

        x = self.decoder(x, context) # (batch_size, target_len, d_model)

        x = x[:, -1, :]  # Shape: (batch_size, d_model)

        logits = self.final_layer(x)

        try:
            # drop the keras mask so it dones't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
    


# ---------------------------------------------------------------------------- #
#                               custom scheduler                               #
# ---------------------------------------------------------------------------- #

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4_000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }