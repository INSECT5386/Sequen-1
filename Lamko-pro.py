import tensorflow as tf
import numpy as np


# =============================
# SwiGLU Layer
# =============================
class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, f_d=8/3):
        super().__init__()
        hidden_dim = int(d_model * f_d + 0.5)  # 반올림
        self.proj = tf.keras.layers.Dense(hidden_dim * 2, use_bias=False, dtype='float32')
        self.out = tf.keras.layers.Dense(d_model, use_bias=False, dtype='float32')

    def call(self, x):
        x_val, x_gate = tf.split(self.proj(x), 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))


# =============================
# Dilated Convolution Layer (Pre-LN)
# =============================
class DilatedConvLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dilation_rate, dropout_rate=0.1):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            dilation_rate=dilation_rate,
            padding='causal',
            use_bias=True,
            kernel_initializer='he_normal',
            dtype='float32'
        )
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        residual = x
        x = self.ln(x)             # Pre-LN: LayerNorm 먼저
        x = self.conv(x)
        x = x + residual           # Residual
        x = self.dropout(x, training=training)
        return x


# =============================
# Sparse Attention Block (Local + Global + Random + Causal)
# =============================
class SparseAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8, window_size=64, global_tokens=4, random_tokens=32, dropout_rate=0.1, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.random_tokens = random_tokens
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

        # QKV 및 출력 프로젝션
        self.qkv_proj = tf.keras.layers.Dense(d_model * 3, use_bias=False, dtype='float32')
        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=False, dtype='float32')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype='float32')

        # 고정된 random indices 미리 생성 (재현성 확보)
        self.fixed_random_indices = self._generate_fixed_random_indices()

    def _generate_fixed_random_indices(self):
        """미리 고정된 random token 인덱스를 생성 (batch마다 동일하게)"""
        # shape: (max_seq_len, random_tokens)
        indices = tf.random.uniform(
            (self.max_seq_len, self.random_tokens),
            minval=0,
            maxval=self.max_seq_len,
            dtype=tf.int32,
            seed=42  # 재현성 위해 고정
        )
        return indices  # (max_seq_len, R)

    def call(self, x, training=False):
        batch_size, seq_len, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        residual = x
        x = self.ln(x)

        # Q, K, V projection
        qkv = self.qkv_proj(x)  # (B, L, 3*d_model)
        q, k, v = tf.split(qkv, 3, axis=-1)  # 각각 (B, L, d_model)

        # Multi-head reshape: (B, L, H, D) -> (B, H, L, D)
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        q = tf.transpose(q, (0, 2, 1, 3))  # (B, H, L, D)
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        # === Create causal mask (L, L) ===
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # lower triangular

        # === 1. Local Attention (with causal) ===
        local_mask = self._create_local_mask(seq_len, causal_mask)
        local_scores = self._compute_attention_scores(q, k, local_mask)

        # === 2. Global Attention (with causal) ===
        global_indices = tf.concat([
            tf.range(self.global_tokens),  # first N
            tf.range(seq_len - self.global_tokens, seq_len)  # last N
        ], axis=0)
        global_mask = self._create_global_mask(seq_len, global_indices, causal_mask)
        global_scores = self._compute_attention_scores(q, k, global_mask)

        # === 3. Random Attention (with causal) ===
        random_mask = self._create_random_mask(seq_len, causal_mask)
        random_scores = self._compute_attention_scores(q, k, random_mask)

        # === Combine all attention scores ===
        combined_scores = local_scores + global_scores + random_scores

        # Softmax + dropout
        attn_weights = tf.nn.softmax(combined_scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # Apply to values
        output = tf.matmul(attn_weights, v)  # (B, H, L, D)
        output = tf.transpose(output, (0, 2, 1, 3))  # (B, L, H, D)
        output = tf.reshape(output, (batch_size, seq_len, self.d_model))

        # Output projection
        output = self.out_proj(output)

        return output + residual  # Residual connection

    def _create_local_mask(self, seq_len, causal_mask):
        indices = tf.range(seq_len)
        distance = tf.abs(tf.expand_dims(indices, 0) - tf.expand_dims(indices, 1))
        mask = tf.cast(distance <= self.window_size, tf.float32)
        return mask * causal_mask  # Apply causal constraint

    def _create_global_mask(self, seq_len, global_indices, causal_mask):
        mask = tf.zeros((seq_len, seq_len), dtype=tf.float32)
        for g in global_indices:
            # Only allow positions i >= g to attend to global token at g
            valid_range = tf.range(g, seq_len)
            update_indices = tf.stack([valid_range, tf.repeat(g, len(valid_range))], axis=1)
            mask = tf.tensor_scatter_nd_update(mask, update_indices, tf.ones_like(valid_range, dtype=tf.float32))
        return mask * causal_mask

    def _create_random_mask(self, seq_len, causal_mask):
        # Use precomputed fixed random indices
        # Shape: (max_seq_len, random_tokens) -> slice to current seq_len
        random_indices = self.fixed_random_indices[:seq_len]  # (seq_len, R)

        # For each position i, we want to know which j's it can attend to (j <= i)
        # We create a mask: for each i, set mask[i][j] = 1 if j is in random_indices[i] AND j <= i
        mask = tf.zeros((seq_len, seq_len), dtype=tf.float32)

        for i in range(seq_len):
            # Get random tokens for this row
            row_random = random_indices[i]  # shape: (R,)
            # Filter only those j where j <= i
            valid_j = tf.boolean_mask(row_random, row_random <= i)
            if tf.size(valid_j) > 0:
                update_indices = tf.stack([tf.repeat(i, tf.size(valid_j)), valid_j], axis=1)
                mask = tf.tensor_scatter_nd_update(mask, update_indices, tf.ones_like(tf.cast(valid_j, tf.float32)))

        return mask * causal_mask

    def _compute_attention_scores(self, q, k, mask):
        """
        q: (B, H, L, D)
        k: (B, H, L, D)
        mask: (L, L)
        """
        # Scaled dot-product
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))  # (B, H, L, L)

        # Expand mask to match scores: (1, 1, L, L)
        mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # (1, 1, L, L)

        # Apply mask
        scores = scores + (1.0 - mask) * -1e9
        return scores


# =============================
# Main Model: Lamko (Decoder-only Language Model)
# =============================
class Lamko(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Embeddings
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model, dtype='float32')
        self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model, dtype='float32')

        # Blocks
        self.blocks = []
        for i in range(n_layers):
            self.blocks.append(DilatedConvLayer(d_model, 2 ** i, dropout_rate))
            if (i + 1) % 3 == 0:
                # Insert Sparse Attention after every 3rd conv block
                self.blocks.append(SparseAttentionBlock(
                    d_model=d_model,
                    num_heads=8,
                    window_size=64,
                    global_tokens=4,
                    random_tokens=32,
                    dropout_rate=dropout_rate,
                    max_seq_len=max_seq_len
                ))
                self.blocks.append(SwiGLU(d_model))
                self.blocks.append(tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype='float32'))

        # Final LayerNorm and output projection (weight tied to token embedding)
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def call(self, x, training=False):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]  # (1, seq_len)

        # Embeddings
        x = self.token_embedding(x) + self.pos_embedding(positions)  # (B, L, D)

        # Pass through blocks
        for block in self.blocks:
            if isinstance(block, SparseAttentionBlock):
                x = block(x, training=training)
            elif isinstance(block, DilatedConvLayer):
                x = block(x, training=training)
            else:
                # SwiGLU or LayerNorm: no training arg needed
                x = block(x)

        # Final normalization
        x = self.ln_f(x)

        # Output logits: weight-tying with token embedding
        logits = tf.matmul(x, self.token_embedding.weights[0], transpose_b=True)

        return logits
