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


import tensorflow as tf

class ParaLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ParaLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.input_dim = None  # 나중에 build에서 설정

    def build(self, input_shape):
        # input_shape: (batch, features) — RNN Cell은 한 timestep만 받음!
        self.input_dim = input_shape[-1]

        # ✅ 입력 차원에 맞춰 가중치 생성
        # W: input → gate (input_dim x units)
        # U: hidden → gate (units x units)

        self.W_i = self.add_weight(shape=(self.input_dim, self.units), initializer='glorot_uniform', name='W_i')
        self.W_f = self.add_weight(shape=(self.input_dim, self.units), initializer='glorot_uniform', name='W_f')
        self.W_o = self.add_weight(shape=(self.input_dim, self.units), initializer='glorot_uniform', name='W_o')
        self.W_c = self.add_weight(shape=(self.input_dim, self.units), initializer='glorot_uniform', name='W_c')

        self.U_i = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_i')
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_f')
        self.U_o = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_o')
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_c')

        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')

        # Layer Normalization
        self.ln_i = tf.keras.layers.LayerNormalization()
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.ln_o = tf.keras.layers.LayerNormalization()
        self.ln_c = tf.keras.layers.LayerNormalization()
        self.ln_cell = tf.keras.layers.LayerNormalization()

        self.built = True

    def call(self, inputs, states):
        h_prev, c_prev = states

        # ✅ 이제 inputs: (batch, input_dim), W: (input_dim, units) → 차원 일치!
        i_t = tf.sigmoid(self.ln_i(tf.matmul(inputs, self.W_i) + tf.matmul(h_prev, self.U_i) + self.b_i))
        f_t = tf.sigmoid(self.ln_f(tf.matmul(inputs, self.W_f) + tf.matmul(h_prev, self.U_f) + self.b_f))
        o_t = tf.sigmoid(self.ln_o(tf.matmul(inputs, self.W_o) + tf.matmul(h_prev, self.U_o) + self.b_o))
        c_hat_t = tf.tanh(self.ln_c(tf.matmul(inputs, self.W_c) + tf.matmul(h_prev, self.U_c) + self.b_c))

        c_t = f_t * c_prev + i_t * c_hat_t
        c_t = self.ln_cell(c_t)
        h_t = o_t * tf.tanh(c_t)

        return h_t, [h_t, c_t]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype)
        ]

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units


# 셀을 RNN 레이어로 감싸기
para_lstm_layer = tf.keras.layers.RNN(
    ParaLSTMCell(units=64),
    return_sequences=True,
    return_state=False,
    name='ParaLSTM'
)

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
