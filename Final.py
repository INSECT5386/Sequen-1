!pip install sentencepiece
import sentencepiece as spm
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
import requests
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

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
limit = 100000
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
from tensorflow.keras import layers

class GroupChannelGate(layers.Layer):
    def __init__(self, d_model, num_groups=4, name="group_channel_gate"):
        super().__init__(name=name)
        self.num_groups = num_groups
        assert d_model % num_groups == 0, "d_model must be divisible by num_groups"
        
        # 각 그룹에 대해 하나의 게이트 출력 — 총 G개 게이트
        self.gate_proj = layers.Dense(num_groups, use_bias=True)

    def call(self, x):
        """
        x: (B, N, D)
        출력: (B, N, D) — 각 그룹별로 독립적인 스칼라 게이트 적용
        """
        B, N, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        G = self.num_groups
        D_per_group = D // G  # 각 그룹의 차원 수

        # (B, N, D) → (B, N, G, D//G)
        x_reshaped = tf.reshape(x, (B, N, G, D_per_group))

        # 게이트 생성: (B, N, G) — 각 그룹에 하나의 스칼라 게이트
        gate = self.gate_proj(x)  # (B, N, G)
        gate = tf.nn.sigmoid(gate)  # (B, N, G) — 0~1 사이 값

        # (B, N, G) → (B, N, G, 1) — 브로드캐스팅 위해 확장
        gate = tf.expand_dims(gate, axis=-1)  # (B, N, G, 1)

        # 게이트 적용: (B, N, G, D//G) * (B, N, G, 1) → (B, N, G, D//G)
        x_gated = x_reshaped * gate

        # 다시 원래 shape으로 복구: (B, N, D)
        return tf.reshape(x_gated, (B, N, D))

class Lo(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.proj = layers.Dense(64, use_bias=True, dtype='float32')
    
    def call(self, x):
        x = self.proj(x)
        x = tf.nn.gelu(x)
        return x

class LoSoU(layers.Layer):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.Q = layers.Dense(64 * num_heads)  # Multi-Query: Q만 확장
        self.K = layers.Dense(64)              # K는 공유 (64차원 고정)
        self.V = Lo(d_model)                   # V는 Lo(MLP) — 튜닝 전용
        self.O = layers.Dense(d_model)         # 출력 복귀

    def call(self, x):
        B, L = tf.shape(x)[0], tf.shape(x)[1]
        
        # Q: Multi-Query — (B, L, 64 * H) → (B, L, H, 64)
        q = self.Q(x)
        q = tf.reshape(q, (B, L, self.num_heads, 64))
        
        # K: 공유 — (B, L, 64)
        k = self.K(x)
        
        # V: Lo(MLP) — (B, L, 64)
        v = self.V(x)
        
        # Element-wise product: Q * K (Broadcasting: K → (B, L, 1, 64))
        # → (B, L, H, 64)
        gate_input = q * tf.expand_dims(k, axis=2)
        
        # Sigmoid → (B, L, H, 64)
        score = tf.nn.sigmoid(gate_input)
        
        # ✅ L1 정규화: 각 위치/헤드에서 64차원 벡터의 합을 1로 만듦
        # axis=-1 (64차원 방향)으로 정규화
        score = score / (tf.reduce_sum(score, axis=-1, keepdims=True) + 1e-8)  # 안정화
        
        # Cumsum along sequence axis (axis=1)
        score = tf.cumsum(score, axis=1)  # (B, L, H, 64)
        
        # V 확장: (B, L, 64) → (B, L, 1, 64) → Broadcasting with score
        v = tf.expand_dims(v, axis=2)     # (B, L, 1, 64)
        
        # Weighted accumulation: score * v → (B, L, H, 64)
        x = score * v
        
        # Concat heads: (B, L, H*64)
        x = tf.reshape(x, (B, L, -1))
        
        # Project back to d_model
        return self.O(x)

class Block(layers.Layer):
    def __init__(self, d_model, num_heads=4, num_groups=8):
        super().__init__()
        self.losou = LoSoU(d_model, num_heads)
        self.group_gate = GroupChannelGate(d_model, num_groups)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.losou(self.norm1(x))      # Token Mixing
        x = x + self.group_gate(self.norm2(x)) # Channel Mixing (Group-wise)
        return x

class Sequen(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model, dtype='float32')
        self.pos_embedding = layers.Embedding(max_seq_len, d_model, dtype='float32')
        
        self.blocks = [Block(d_model) for _ in range(n_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def call(self, x, training=False):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]  # (1, seq_len)

        x = self.token_embedding(x) + self.pos_embedding(positions)  # (batch, seq_len, d_model)

        for block in self.blocks:
            x = block(x)
            
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
    model = Sequen(vocab_size, max_seq_len=max_len, d_model=256, n_layers=9, dropout_rate=0.1)
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
sample_text = generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.75, min_len=20)
print("\n===== 생성 결과 =====\n")
print(sample_text)

prompt = "딥러닝에 대해 설명하세요."
sample_text = generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.75, min_len=20)
print("\n===== 생성 결과 =====\n")
print(sample_text)

prompt = "안녕하세요."
sample_text = generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.75, min_len=20)
print("\n===== 생성 결과 =====\n")
print(sample_text)

prompt = "안녕하세요."
sample_text = generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.75, min_len=20)
print("\n===== 생성 결과 =====\n")
print(sample_text)

prompt = "오늘의 날씨는 어떤가요?."
sample_text = generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.75, min_len=20)
print("\n===== 생성 결과 =====\n")
print(sample_text)

prompt = "오늘의 날씨는 어떤가요?."
sample_text = generate_text_topp(model, prompt, max_len=96, max_gen=96, p=0.9, temperature=0.75, min_len=20)
print("\n===== 생성 결과 =====\n")
print(sample_text)
