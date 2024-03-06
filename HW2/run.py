import numpy as np
import tensorflow as tf
import time
import pandas as pd
from collections import Counter



# Hyperparameters
np.random.seed(1234)
tf.random.set_seed(1234)
vocabulary_size = 1024
embedding_dim = 64
max_length = 45  # Sentences are padded to this length
size_input = max_length * embedding_dim
size_hidden1 = 128
size_hidden2 = 128
size_hidden3 = 128
size_hidden4 = 128
size_output = 2  # For binary classification: positive or neutral
learning_rate = 1e-4
NUM_EPOCHS = 100
batch_size = 32

class MLP(object):
    def __init__(self):
        self.device = 'gpu'  # Adjust if necessary
        self.E = tf.Variable(tf.random.uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
        self.W1 = tf.Variable(tf.random.normal([size_input, size_hidden1], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([size_hidden1]))
        self.W2 = tf.Variable(tf.random.normal([size_hidden1, size_hidden2], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([size_hidden2]))
        self.W3 = tf.Variable(tf.random.normal([size_hidden2, size_hidden3], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([size_hidden3]))
        self.W4 = tf.Variable(tf.random.normal([size_hidden3, size_hidden4], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([size_hidden4]))
        self.W5 = tf.Variable(tf.random.normal([size_hidden4, size_output], stddev=0.1))
        self.b5 = tf.Variable(tf.zeros([size_output]))
        self.variables = [self.E, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5]

    def forward(self, X):
        # Embedding layer
        embedded = np.dot(X, self.E.numpy())  # One-hot encoded X multiplied by the embedding matrix
        
        # ReLU activation functions
        z1 = np.maximum(0, np.dot(embedded, self.W1.numpy()) + self.b1.numpy())
        z2 = np.maximum(0, np.dot(z1, self.W2.numpy()) + self.b2.numpy())
        z3 = np.maximum(0, np.dot(z2, self.W3.numpy()) + self.b3.numpy())
        z4 = np.maximum(0, np.dot(z3, self.W4.numpy()) + self.b4.numpy())
        
        # Logits calculation
        logits = np.dot(z4, self.W5.numpy()) + self.b5.numpy()
        return logits

    def loss(self, y_pred, y_true):
        # Compute softmax
        exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # One-hot encoding
        one_hot = np.zeros_like(softmax)
        one_hot[np.arange(len(y_true)), y_true] = 1
        
        # Compute cross-entropy loss
        loss = -np.mean(one_hot * np.log(softmax + 1e-10))  # Add a small epsilon to avoid numerical instability
        
        return loss
    def backward(self, X_train, y_train):
        # Forward pass
        logits = self.forward(X_train)
        
        # Compute gradients of loss with respect to weights and biases
        d_loss_d_logits = self.loss_derivative(logits, y_train)
        d_loss_d_W5 = np.dot(self.z4.T, d_loss_d_logits)
        d_loss_d_b5 = np.sum(d_loss_d_logits, axis=0)
        d_loss_d_z4 = np.dot(d_loss_d_logits, self.W5.T)
        d_loss_d_a4 = np.where(self.z4 > 0, d_loss_d_z4, 0)  # ReLU derivative
        d_loss_d_W4 = np.dot(self.z3.T, d_loss_d_a4)
        d_loss_d_b4 = np.sum(d_loss_d_a4, axis=0)
        d_loss_d_z3 = np.dot(d_loss_d_a4, self.W4.T)
        d_loss_d_a3 = np.where(self.z3 > 0, d_loss_d_z3, 0)  # ReLU derivative
        d_loss_d_W3 = np.dot(self.z2.T, d_loss_d_a3)
        d_loss_d_b3 = np.sum(d_loss_d_a3, axis=0)
        d_loss_d_z2 = np.dot(d_loss_d_a3, self.W3.T)
        d_loss_d_a2 = np.where(self.z2 > 0, d_loss_d_z2, 0)  # ReLU derivative
        d_loss_d_W2 = np.dot(self.z1.T, d_loss_d_a2)
        d_loss_d_b2 = np.sum(d_loss_d_a2, axis=0)
        d_loss_d_z1 = np.dot(d_loss_d_a2, self.W2.T)
        d_loss_d_a1 = np.where(self.z1 > 0, d_loss_d_z1, 0)  # ReLU derivative
        d_loss_d_W1 = np.dot(self.flat_embedded.T, d_loss_d_a1)
        d_loss_d_b1 = np.sum(d_loss_d_a1, axis=0)
        d_loss_d_embedded = np.dot(d_loss_d_a1, self.W1.T)
        d_loss_d_E = np.dot(X_train.T, d_loss_d_embedded)
        
        # Update weights and biases
        self.W5 -= learning_rate * d_loss_d_W5
        self.b5 -= learning_rate * d_loss_d_b5
        self.W4 -= learning_rate * d_loss_d_W4
        self.b4 -= learning_rate * d_loss_d_b4
        self.W3 -= learning_rate * d_loss_d_W3
        self.b3 -= learning_rate * d_loss_d_b3
        self.W2 -= learning_rate * d_loss_d_W2
        self.b2 -= learning_rate * d_loss_d_b2
        self.W1 -= learning_rate * d_loss_d_W1
        self.b1 -= learning_rate * d_loss_d_b1
        self.E -= learning_rate * d_loss_d_E
        
        return loss_value

    def compute_output(self, X):
        # Forward pass
        logits = self.forward(X)
        
        # Softmax activation
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs


# Load your data
sentences = pd.read_csv('final_processed_sentences.csv', header=None)
sentiments = pd.read_csv('sentiments.csv', header=None)

# Calculate split indices
total_samples = sentences.shape[0]
train_end = int(total_samples * 0.6)
val_end = int(total_samples * 0.8)

# Manually split the datasets
X_train = sentences.iloc[:train_end, 0].astype(str)  # Ensure X_train sentences are strings
y_train = sentiments.iloc[:train_end, 0]

X_val = sentences.iloc[train_end:val_end, 0].astype(str)
y_val = sentiments.iloc[train_end:val_end, 0]

X_test = sentences.iloc[val_end:, 0].astype(str)
y_test = sentiments.iloc[val_end:, 0]

# Tokenize sentences and filter out '<pad>'
all_words = [word for sentence in X_train for word in sentence.split() if word != "<pad>"]

# Count the frequency of each word
word_freq = Counter(all_words)

# Get most common words
most_common_words = word_freq.most_common(vocabulary_size)

# Extract just the words, excluding their counts
vocab = [word for word, count in most_common_words]

print(f"Size of vocabulary: {len(vocab)}")

for x in range(100):
    print(most_common_words[x])

def preprocess_input(X, vocab, max_length, vocabulary_size):
    processed_X = []
    for sentence in X:
        tokens = sentence.split()
        one_hot_sentence = [np.eye(vocabulary_size)[vocab.index(word)] for word in tokens if word in vocab]
        if len(one_hot_sentence) < max_length:
            pad_length = max_length - len(one_hot_sentence)
            one_hot_sentence += [np.zeros(vocabulary_size)] * pad_length
        processed_X.append(one_hot_sentence)
    return np.array(processed_X)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
mlp_on_cpu = MLP()

time_start = time.time()

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    
    # Training loop
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Tokenize and preprocess input data
        X_batch_processed = preprocess_input(X_batch, vocab, max_length, vocabulary_size)
        
        with tf.GradientTape() as tape:
            logits = mlp_on_cpu.forward(X_batch_processed)
            loss_value = mlp_on_cpu.loss(logits, y_batch)
        
        gradients = tape.gradient(loss_value, mlp_on_cpu.variables)
        optimizer.apply_gradients(zip(gradients, mlp_on_cpu.variables))
        
        total_loss += loss_value.numpy()
    
    # Compute average training loss for the epoch
    average_train_loss = total_loss / (len(X_train) / batch_size)
    
    # Validation loop
    val_total_loss = 0.0
    val_correct = 0
    
    for i in range(0, len(X_val), batch_size):
        X_val_batch = X_val[i:i+batch_size]
        y_val_batch = y_val[i:i+batch_size]
        
        # Tokenize and preprocess validation data
        X_val_batch_processed = preprocess_input(X_val_batch, vocab, max_length, vocabulary_size)
        
        val_logits = mlp_on_cpu.forward(X_val_batch_processed)
        val_loss_value = mlp_on_cpu.loss(val_logits, y_val_batch)
        
        val_total_loss += val_loss_value.numpy()
        
        # Compute accuracy for validation set
        val_predictions = np.argmax(mlp_on_cpu.compute_output(X_val_batch_processed), axis=1)
        val_correct += np.sum(val_predictions == y_val_batch.numpy())
    
    # Compute average validation loss and accuracy
    average_val_loss = val_total_loss / (len(X_val) / batch_size)
    val_accuracy = val_correct / len(X_val)
    
    # Print progress
    print(f'Epoch {epoch + 1}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

time_taken = time.time() - time_start
print(f'Training completed in {time_taken:.2f} seconds.')
