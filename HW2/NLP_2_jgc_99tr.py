import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import CategoricalAccuracy

# Instantiate metrics
train_accuracy = CategoricalAccuracy()
val_accuracy = CategoricalAccuracy()

# Ensure reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

# Load data
final_processed_sentences = pd.read_csv('final_processed_sentences.csv', header=None)
sentiments = pd.read_csv('sentiments.csv', header=None)

# Assuming sentiments are in the 'sentiments' DataFrame and it contains 'positive' and 'negative' labels
# Map 'positive' to 1 and 'negative' to 0
sentiments = sentiments[0].map({'positive': 1, 'negative': 0})

print(sentiments)


# Assuming the CSV files contain the appropriate data without headers
y = sentiments.values.flatten()



# Assuming final_processed_sentences is your input data
texts = final_processed_sentences[0].tolist()  # Convert DataFrame column to list

# Tokenize the text
tokenizer = Tokenizer(num_words=10000)  # Adjust num_words based on your vocabulary size
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to ensure uniform input size
X = pad_sequences(sequences, maxlen=45)  # Adjust maxlen based on your needs


# Split data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=1234)  # 0.25 x 0.8 = 0.2

# Convert labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

class MLP(object):
    def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_hidden5, size_output, device=None):
        """
        Initializes the model's layers and variables.
        """
        self.size_input, self.size_hidden1, self.size_hidden2, self.size_hidden3, self.size_hidden4, self.size_hidden5, self.size_output, self.device =\
        size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_hidden5, size_output, device
        
        # Initialize weights and biases for the model layers
        self.W1 = tf.Variable(tf.random.normal([self.size_input, self.size_hidden1], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([1, self.size_hidden1]))
        self.W2 = tf.Variable(tf.random.normal([self.size_hidden1, self.size_hidden2], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([1, self.size_hidden2]))
        self.W3 = tf.Variable(tf.random.normal([self.size_hidden2, self.size_hidden3], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([1, self.size_hidden3]))
        self.W4 = tf.Variable(tf.random.normal([self.size_hidden3, self.size_hidden4], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([1, self.size_hidden4]))
        self.W5 = tf.Variable(tf.random.normal([self.size_hidden4, self.size_hidden5], stddev=0.1))
        self.b5 = tf.Variable(tf.zeros([1, self.size_hidden5]))
        self.W6 = tf.Variable(tf.random.normal([self.size_hidden5, self.size_output], stddev=0.1))
        self.b6 = tf.Variable(tf.zeros([1, self.size_output]))
        
        # Collect all variables
        self.variables = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5, self.W6, self.b6]
    
    def forward(self, X):
        """
        Performs the forward pass through the model.
        """
        X = tf.cast(X, dtype=tf.float32)  # Ensure X is float32
        # Input to first hidden layer
        h1 = tf.matmul(X, self.W1) + self.b1
        z1 = tf.nn.relu(h1)
        
        # First hidden layer to second hidden layer
        h2 = tf.matmul(z1, self.W2) + self.b2
        z2 = tf.nn.relu(h2)
        
        # Second hidden layer to third hidden layer
        h3 = tf.matmul(z2, self.W3) + self.b3
        z3 = tf.nn.relu(h3)

        # Third hidden layer to fourth hidden layer
        h4 = tf.matmul(z3, self.W4) + self.b4
        z4 = tf.nn.relu(h4)

        # Fourth hidden layer to fifth hidden layer
        h5 = tf.matmul(z4, self.W5) + self.b5
        z5 = tf.nn.relu(h5)
        
        # Fifth hidden layer to output
        output = tf.matmul(z5, self.W6) + self.b6
        return output

    
    def loss(self, y_pred, y_true):
        """
        Computes the loss using categorical crossentropy.
        """
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_tf, y_pred_tf, from_logits=True))
    
    def backward(self, X_train, y_train):
        """
        Performs the backward pass and updates the model's variables.
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        return grads

# Model configuration
size_input = X_train.shape[1]
size_hidden1 = 128
size_hidden2 = 128
size_hidden3 = 128
size_hidden4 = 128
size_hidden5 = 128
size_output = 2  
device = 'gpu'

# Instantiate the model
mlp_model = MLP(size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_hidden5, size_output, device)

# Training configuration
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
NUM_EPOCHS = 1000

# Training loop
time_start = time.time()

# Parameters
batch_size = 2048  # You can adjust this size as needed

# Convert the training and validation data into tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


for epoch in range(NUM_EPOCHS):
    train_accuracy.reset_states()
    val_accuracy.reset_states()
    
    # Initialize variables to accumulate losses
    total_train_loss = 0
    total_val_loss = 0
    train_batches = 0
    val_batches = 0

    # Training loop - iterating over batches
    for X_batch, y_batch in train_dataset:
        grads = mlp_model.backward(X_batch, y_batch)
        optimizer.apply_gradients(zip(grads, mlp_model.variables))
        
        # Update training metric and loss
        train_predictions = mlp_model.forward(X_batch)
        batch_train_loss = mlp_model.loss(train_predictions, y_batch).numpy()
        
        total_train_loss += batch_train_loss
        train_batches += 1
        train_accuracy.update_state(y_batch, train_predictions)
    
    # Validation loop - iterating over batches
    for X_batch, y_batch in val_dataset:
        val_predictions = mlp_model.forward(X_batch)
        batch_val_loss = mlp_model.loss(val_predictions, y_batch).numpy()
        
        total_val_loss += batch_val_loss
        val_batches += 1
        val_accuracy.update_state(y_batch, val_predictions)
    
    # Calculate average loss over all batches
    avg_train_loss = total_train_loss / train_batches
    avg_val_loss = total_val_loss / val_batches

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy.result().numpy():.4f}, Val Acc: {val_accuracy.result().numpy():.4f}")

time_taken = time.time() - time_start
print(f"Training completed in {time_taken:.2f} seconds.")

