import tensorflow as tf
import numpy as np

class NPLM(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size=2):
        super().__init__()
        self.context_size = context_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.hidden = tf.keras.layers.Dense(hidden_dim, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        embeds = self.embedding(x)
        flat = tf.reshape(embeds, (-1, self.context_size * embeds.shape[-1]))
        h = self.hidden(flat)
        return self.output_layer(h)
vocab_size = 10
context_size = 2
embed_dim = 16
hidden_dim = 32

X = []
y = []
for a in range(vocab_size):
    for b in range(vocab_size):
        X.append([a, b])
        y.append(b)

X = np.array(X)
y = np.array(y)

model = NPLM(vocab_size, embed_dim, hidden_dim, context_size)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-2)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
history = model.fit(X, y, batch_size=16, epochs=20, verbose=1)

test_input = np.array([[2, 7], [9, 3], [4, 4]])
pred_logits = model.predict(test_input)
preds = tf.argmax(pred_logits, axis=-1).numpy()

print("test input:", test_input)
print("Model outputs:", preds)
