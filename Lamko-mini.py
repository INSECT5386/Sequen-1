!pip install sentencepiece
import sentencepiece as spm
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
import requests

print("1")

# 시드 고정
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# GPU/CPU 전략
strategy = tf.distribute.get_strategy()
print("✅ GPU/CPU 전략 사용")

# Mixed precision OFF
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_global_policy(policy)
print("✅ Mixed precision OFF, float32 연산")

# =======================
# 파일 다운로드
# =======================
def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")

DATA_PATH = "converted.jsonl"
TOKENIZER_PATH = "ko_unigram.model"

if not os.path.exists(DATA_PATH):
    download_file(
        "https://huggingface.co/datasets/Yuchan5386/SFT/resolve/main/data_shuffled_1.jsonl?download=true",
        DATA_PATH
    )

if not os.path.exists(TOKENIZER_PATH):
    download_file(
        "https://huggingface.co/Yuchan5386/inlam-100m/resolve/main/ko_unigram.model?download=true",
        TOKENIZER_PATH
    )

sp = spm.SentencePieceProcessor(TOKENIZER_PATH)
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>")
sep_id = sp.piece_to_id("<sep>")
end_id = sp.piece_to_id("<end>")
unk_id = sp.piece_to_id("<unk>")
vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")
limit = 50000
max_len = 96
batch_size = 32  # GPU 메모리 절약용
print(limit // batch_size)

def text_to_ids(text):
    return sp.encode(text, out_type=int)

def ids_to_text(ids):
    return sp.decode(ids)

# Lamko.weights (1).h5


# =======================
# 데이터셋
# =======================
def jsonl_stream(file_path, limit=None, sample_rate=1.0):
    import random
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue
            data = json.loads(line)
            conversations = data.get("conversations", [])
            for i in range(0, len(conversations) - 1, 2):
                human_msg = conversations[i]
                gpt_msg   = conversations[i + 1]
                if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                    continue
                prompt   = human_msg.get("value", "").strip()
                response = gpt_msg.get("value", "").strip()
                full = f"<start> {prompt} <sep> {response} <end>"
                if "<sep>" not in full:
                    continue
                sep_index  = full.index("<sep>")
                input_text = full[:sep_index + len("<sep>")].strip()
                target_text = full[sep_index + len("<sep>"):].strip()

                input_ids  = text_to_ids(input_text)
                target_ids = text_to_ids(target_text + " <end>")

                available_len = max_len - len(input_ids)
                if available_len <= 0:
                    input_ids = input_ids[-max_len:]
                    target_ids = []
                    target_mask = [0] * len(input_ids)
                else:
                    target_ids = target_ids[:available_len]
                    target_mask = [0] * len(input_ids) + [1] * len(target_ids)

                full_input = input_ids + target_ids
                pad_len = max_len - len(full_input)
                full_input += [pad_id] * pad_len
                target_mask += [0] * pad_len

                target_seq = full_input[1:] + [end_id]
                target_seq = target_seq[:max_len]

                masked_target = [
                    t if m == 1 else pad_id
                    for t, m in zip(target_seq, target_mask)
                ]

                yield (
                    tf.convert_to_tensor(full_input, dtype=tf.int32),
                    tf.convert_to_tensor(masked_target, dtype=tf.int32)
                )

                count += 1
                if limit is not None and count >= limit:
                    return

dataset = tf.data.Dataset.from_generator(
    lambda: jsonl_stream(DATA_PATH, limit=limit),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
    ),
)
dataset = dataset.shuffle(1000, seed=SEED).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
dist_dataset = strategy.experimental_distribute_dataset(dataset)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras import layers
import math

class GatedLinearAttention(layers.Layer):
    def __init__(self, d_model, causal=True, gate_fn='swish', **kwargs):
        super(GatedLinearAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.causal = causal  # True로 고정 (요구사항)
        self.gate_fn_name = gate_fn

        # gating function 선택
        if gate_fn == 'swish':
            self.gate_fn = lambda x: x * tf.sigmoid(x)
        elif gate_fn == 'relu':
            self.gate_fn = tf.nn.relu
        elif gate_fn == 'sigmoid':
            self.gate_fn = tf.sigmoid
        else:
            raise ValueError(f"Unsupported gate_fn: {gate_fn}")

        # Q, K, V projection (1x1 convolution = dense per token)
        self.Wq = layers.Dense(d_model, use_bias=False, name='Wq')
        self.Wk = layers.Dense(d_model, use_bias=False, name='Wk')
        self.Wv = layers.Dense(d_model, use_bias=False, name='Wv')
        self.Wo = layers.Dense(d_model, use_bias=False, name='Wo')  # output projection

    def call(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) — optional, 1 for real token, 0 for padding
        """
        Q = self.Wq(x)  # (B, L, D)
        K = self.Wk(x)  # (B, L, D)
        V = self.Wv(x)  # (B, L, D)

        # Apply gating
        Q_gated = self.gate_fn(Q)      # (B, L, D)
        K_gated = self.gate_fn(K)      # (B, L, D)

        # Compute: K_gated ⊙ V
        KV = K_gated * V               # (B, L, D)

        # Apply causal cumsum (axis=1: along sequence)
        # cumsum naturally respects causality: position i only sees 0..i
        KV_cumsum = tf.cumsum(KV, axis=1)  # (B, L, D)

        # Apply Q_gated ⊙ KV_cumsum
        output = Q_gated * KV_cumsum   # (B, L, D)

        # Optional: apply mask (set padded positions to 0)
        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], dtype=output.dtype)  # (B, L, 1)
            output *= mask

        # Final output projection
        output = self.Wo(output)       # (B, L, D)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "causal": self.causal,
            "gate_fn": self.gate_fn_name,
        })
        return config

class SwiGLU(layers.Layer):
    def __init__(self, d_model, f_d=8/3):
        super().__init__()
        hidden_dim = int(d_model * f_d + 0.5)  # 반올림
        self.proj = layers.Dense(hidden_dim * 2, use_bias=True, dtype='float32')
        self.out = layers.Dense(d_model, use_bias=True, dtype='float32')

    def call(self, x):
        x_val, x_gate = tf.split(self.proj(x), 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))

class Adapter(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.proj = layers.Dense(128, use_bias=True, dtype='float32')
        self.out = layers.Dense(d_model, use_bias=True, dtype='float32')

    def call(self, x):
        x_val, x_gate = tf.split(self.proj(x), 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))
        
class RNNa(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model, dtype='float32')
        self.pos_embedding = layers.Embedding(max_seq_len, d_model, dtype='float32')
        self.block_1 = GatedLinearAttention(d_model)
        self.block_2 = GatedLinearAttention(d_model)

        self.adapter_1 = Adapter(d_model=d_model)
        self.adapter = Adapter(d_model=d_model)
        
        self.ffn_1 = SwiGLU(d_model=d_model)
        self.ffn_2 = SwiGLU(d_model=d_model)
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def call(self, x, training=False):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]  # (1, seq_len)

        x = self.token_embedding(x) + self.pos_embedding(positions)  # (batch, seq_len, d_model)

        x = self.block_1(x, training=training)
        x = self.ffn_1(x)
        x = self.adapter(x)
        x = self.block_2(x, training=training)
        x = self.ffn_2(x)
        x = self.adapter_1(x)
    
        x = self.ln_f(x)  # (batch, seq_len, d_model)

        # ✅ 수정: 안전하게 embedding matrix 참조
        embedding_matrix = self.token_embedding.embeddings
        logits = tf.matmul(x, embedding_matrix, transpose_b=True)  # (batch, seq_len, vocab_size)

        return logits

def smoothed_loss_keras(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1) * mask
    return tf.reduce_sum(per_tok) / (tf.reduce_sum(mask) + 1e-8)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    pred_id = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    acc = tf.cast(tf.equal(y_true, pred_id), tf.float32) * mask
    return tf.reduce_sum(acc) / (tf.reduce_sum(mask) + 1e-8)
# =======================
# 모델 생성 & 학습
# =======================
with strategy.scope():
    model = RNNa(vocab_size, max_seq_len=max_len, d_model=256, n_layers=9, dropout_rate=0.1)
    dummy_input = tf.zeros((batch_size, max_len), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-8, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=smoothed_loss_keras, metrics=[masked_accuracy])
    history = model.fit(dist_dataset, epochs=1, verbose=1)

# =======================
# 가중치 저장
# =======================
model.save_weights("RNNa.weights.h5")
print("✅ 모델 가중치 저장 완료!")

# =======================
def generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.68, min_len=20):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    
    for step in range(max_gen):
        input_seq = generated[-max_len:] if len(generated) > max_len else generated
        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded], dtype=tf.int32)
        
        logits = model(input_tensor, training=False).numpy()[0, len(input_seq)-1]
        logits[end_id] -= 5.0
        logits[pad_id] -= 10.0
        
        probs = tf.nn.softmax(logits / temperature).numpy()
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, p)
        top_idx = sorted_idx[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1] / sorted_probs[:cutoff + 1].sum()
        
        next_token = int(np.random.choice(top_idx, p=top_probs))
        if next_token == end_id and len(generated) >= min_len:
            break
        generated.append(next_token)
    
    return ids_to_text(generated)

# =======================
# 테스트 생성
# =======================
prompt = "딥러닝에 대해 설명하세요."
sample_text = generate_text_topp(model, prompt, p=0.9)
print("\n===== 생성 결과 =====\n")
print(sample_text)
