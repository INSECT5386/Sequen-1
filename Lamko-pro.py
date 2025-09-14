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
batch_size = 48  # GPU 메모리 절약용
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

# 1. Gated Linear Unit with Cross-Position Mixing
class GLUMixer(layers.Layer):
    def __init__(self, d_model, expansion_factor=2):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * expansion_factor
        
        # Position mixing weights (learnable)
        self.pos_mix = layers.Dense(d_model, use_bias=False, dtype='float32')
        
        # GLU components
        self.gate_proj = layers.Dense(self.hidden_dim, use_bias=False, dtype='float32')
        self.value_proj = layers.Dense(self.hidden_dim, use_bias=False, dtype='float32')
        self.out_proj = layers.Dense(d_model, use_bias=False, dtype='float32')
        
    def call(self, x):
        batch_size, seq_len, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        # Position-wise mixing: 각 토큰이 이웃 토큰들과 상호작용
        # Circular shift를 통한 토큰 간 정보 교환
        x_shifted_left = tf.concat([x[:, 1:, :], x[:, :1, :]], axis=1)
        x_shifted_right = tf.concat([x[:, -1:, :], x[:, :-1, :]], axis=1)
        
        # 위치별 가중 합성
        x_mixed = self.pos_mix(x + 0.1 * x_shifted_left + 0.1 * x_shifted_right)
        
        # GLU 변환
        gate = tf.nn.sigmoid(self.gate_proj(x_mixed))
        value = self.value_proj(x_mixed)
        
        return self.out_proj(gate * value)


# 2. Fourier Transform Based Mixing (Global Receptive Field)
class FourierMixer(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = tf.Variable(1.0, trainable=True, dtype='float32')
        self.proj_in = layers.Dense(d_model, use_bias=False, dtype='float32')
        self.proj_out = layers.Dense(d_model, use_bias=False, dtype='float32')
        
    def call(self, x):
        # 입력 변환
        x = self.proj_in(x)
        
        # FFT를 통한 주파수 도메인 처리 (global mixing)
        x_freq = tf.signal.fft(tf.cast(x, tf.complex64))
        
        # 학습 가능한 스케일링
        x_freq = x_freq * tf.cast(self.scale, tf.complex64)
        
        # 역변환
        x = tf.math.real(tf.signal.ifft(x_freq))
        
        return self.proj_out(x)


# 3. Rotation-Based Position Interaction
class RotaryMixer(layers.Layer):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        
        # RoPE-style rotation matrices
        dim = d_model // 2
        inv_freq = 1.0 / (10000 ** (tf.cast(tf.range(0, dim, 2), tf.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self.q_proj = layers.Dense(d_model, use_bias=False, dtype='float32')
        self.k_proj = layers.Dense(d_model, use_bias=False, dtype='float32')
        self.v_proj = layers.Dense(d_model, use_bias=False, dtype='float32')
        self.out_proj = layers.Dense(d_model, use_bias=False, dtype='float32')
        
    def rotate_half(self, x):
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([-x2, x1], axis=-1)
    
    def apply_rotary_pos_emb(self, x, pos):
        # Rotary positional embedding
        cos_pos = tf.cos(pos)
        sin_pos = tf.sin(pos)
        return x * cos_pos + self.rotate_half(x) * sin_pos
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        
        # Position encoding
        positions = tf.cast(tf.range(seq_len), tf.float32)[:, None]
        freqs = positions * self.inv_freq[None, :]
        pos_emb = tf.concat([freqs, freqs], axis=-1)
        
        # Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply rotary embeddings
        q = self.apply_rotary_pos_emb(q, pos_emb)
        k = self.apply_rotary_pos_emb(k, pos_emb)
        
        # Simplified interaction (no full attention)
        # Local-global mixing through element-wise operations
        interaction = q * tf.reduce_mean(k, axis=1, keepdims=True) + k * tf.reduce_mean(q, axis=1, keepdims=True)
        output = interaction * v
        
        return self.out_proj(output)


# 4. Depthwise Separable Convolution with Large Kernels
class LargeKernelConv(layers.Layer):
    def __init__(self, d_model, kernel_size=15, dropout_rate=0.1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            padding='causal',
            use_bias=False,
            dtype='float32'
        )
        
        # Pointwise convolution
        self.pointwise_conv = layers.Conv1D(
            filters=d_model,
            kernel_size=1,
            use_bias=False,
            dtype='float32'
        )
        
        self.ln = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, x, training=False):
        residual = x
        x = self.ln(x)
        x = self.depthwise_conv(x)
        x = tf.nn.gelu(x)
        x = self.pointwise_conv(x)
        x = x + residual
        return self.dropout(x, training=training)

class DilatedConvLayer(layers.Layer):
    def __init__(self, d_model, dilation_rate, dropout_rate=0.1):
        super().__init__()
        self.conv = layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            dilation_rate=dilation_rate,
            padding='causal',
            use_bias=True,
            kernel_initializer='he_normal',
            dtype='float32'
        )
        self.ln = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        residual = x
        x = self.ln(x)             # ← Pre-LN: LN 먼저
        x = self.conv(x)
        x = x + residual           # ← Residual
        x = self.dropout(x, training=training)
        return x
# 5. Enhanced Lamko with Token Interaction Layers
class Lamko(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model, dtype='float32')
        self.pos_embedding = layers.Embedding(max_seq_len, d_model, dtype='float32')
        
        self.blocks = []
        for i in range(n_layers):
            # Dilated conv for local patterns
            self.blocks.append(DilatedConvLayer(d_model, 2 ** (i % 6), dropout_rate))
            
            # Add token interaction layers periodically
            if (i + 1) % 2 == 0:
                # 다양한 상호작용 방식을 번갈아 사용
                if i % 4 == 1:
                    self.blocks.append(GLUMixer(d_model))
                elif i % 4 == 3:
                    self.blocks.append(FourierMixer(d_model))
                    
            # SwiGLU every 3 layers
            if (i + 1) % 3 == 0:
                self.blocks.append(SwiGLU(d_model))
                self.blocks.append(layers.LayerNormalization(epsilon=1e-5, dtype='float32'))
                
        # Add final interaction layer
        self.blocks.append(RotaryMixer(d_model, max_seq_len))
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        
    def call(self, x, training=False):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]
        
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        for block in self.blocks:
            if isinstance(block, (SwiGLU, GLUMixer, FourierMixer, RotaryMixer)):
                x = x + block(x)  # Residual connection for mixing layers
            else:
                x = block(x, training=training) if hasattr(block, 'training') else block(x)
                
        x = self.ln_f(x)
        logits = tf.matmul(x, self.token_embedding.weights[0], transpose_b=True)
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
    model = Lamko(vocab_size, max_seq_len=max_len, d_model=512, n_layers=9, dropout_rate=0.1)
    dummy_input = tf.zeros((batch_size, max_len), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-8, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=smoothed_loss_keras, metrics=[masked_accuracy])
    history = model.fit(dist_dataset, epochs=1, verbose=1)

# =======================
# 가중치 저장
# =======================
model.save_weights("Lamko.weights.h5")
print("✅ 모델 가중치 저장 완료!")



# =======================
def generate_text_topp(model, prompt, max_len=1024, max_gen=200, p=0.9, temperature=0.68, min_len=20):
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
