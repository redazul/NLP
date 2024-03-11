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
import csv
from sklearn.metrics import confusion_matrix

# Instantiate metrics
train_accuracy = CategoricalAccuracy()
val_accuracy = CategoricalAccuracy()

# Load data
final_processed_sentences = pd.read_csv('final_processed_sentences.csv', header=None)
sentiments = pd.read_csv('sentiments.csv', header=None)

# Assuming sentiments are in the 'sentiments' DataFrame and it contains 'positive' and 'negative' labels
# Map 'positive' to 1 and 'negative' to 0
sentiments = sentiments[0].map({'positive': 1, 'negative': 0}).values

# Assuming the CSV files contain the appropriate data without headers
y = sentiments.flatten()

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
    def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_hidden5, size_output, vocab_size, embedding_dim, device=None):
        """
        Initializes the model's layers, variables, and the embedding matrix.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device

        # Embedding matrix
        self.EmbeddingMatrix = tf.Variable(tf.random.uniform([vocab_size, embedding_dim], -1.0, 1.0))

        # Initialize weights and biases for the model layers following the embedding lookup
        self.W1 = tf.Variable(tf.random.normal([size_input * embedding_dim, size_hidden1], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([1, size_hidden1]))
        self.W2 = tf.Variable(tf.random.normal([size_hidden1, size_hidden2], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([1, size_hidden2]))
        self.W3 = tf.Variable(tf.random.normal([size_hidden2, size_hidden3], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([1, size_hidden3]))
        self.W4 = tf.Variable(tf.random.normal([size_hidden3, size_hidden4], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([1, size_hidden4]))
        self.W5 = tf.Variable(tf.random.normal([size_hidden4, size_hidden5], stddev=0.1))
        self.b5 = tf.Variable(tf.zeros([1, size_hidden5]))
        self.W6 = tf.Variable(tf.random.normal([size_hidden5, size_output], stddev=0.1))
        self.b6 = tf.Variable(tf.zeros([1, size_output]))
        
        # Collect all variables including the embedding matrix
        self.variables = [self.EmbeddingMatrix, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5, self.W6, self.b6]

    def forward(self, X):
        """
        Performs the forward pass through the model.
        """
        # Lookup embeddings for each word in the input sequence
        X_embedded = tf.nn.embedding_lookup(self.EmbeddingMatrix, X)

        # Flatten the embeddings to pass through the dense layers
        X_flattened = tf.reshape(X_embedded, [tf.shape(X)[0], -1])

        # Input to first hidden layer
        h1 = tf.matmul(X_flattened, self.W1) + self.b1
        z1 = tf.nn.relu(h1)
        
        # Proceed through remaining hidden layers...
        h2 = tf.matmul(z1, self.W2) + self.b2
        z2 = tf.nn.relu(h2)
        
        h3 = tf.matmul(z2, self.W3) + self.b3
        z3 = tf.nn.relu(h3)

        h4 = tf.matmul(z3, self.W4) + self.b4
        z4 = tf.nn.relu(h4)

        h5 = tf.matmul(z4, self.W5) + self.b5
        z5 = tf.nn.relu(h5)
        
        # Output layer
        output = tf.matmul(z5, self.W6) + self.b6
        return output
    
    def loss(self, y_pred, y_true, l2_lambda=0.01):
        """
        Computes the loss using categorical crossentropy and adds L2 regularization.
        """
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        base_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_tf, y_pred_tf, from_logits=True))
        
        # L2 Regularization for the weights
        l2_losses = [tf.nn.l2_loss(v) for v in self.variables if 'W' in v.name]  # Assuming weight variables have 'W' in their name
        if l2_losses:  # Check if the list is not empty
            l2_loss = tf.add_n(l2_losses) * l2_lambda
        else:
            l2_loss = 0.0
        
        return base_loss + l2_loss
    
    def backward(self, X_train, y_train):
        """
        Performs the backward pass and updates the model's variables.
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        return grads



# This function is added to extract the actual predictions from the model's output logits
def extract_predictions(logits):
    return tf.argmax(logits, axis=1).numpy()



# vocabulary size 
vocab_size = 10000

#Best Performance Hyper Params
embedding_dim = 32  # The output dimension for your embedding layer
size_input = 45
size_hidden1 = 32
size_hidden2 = 16
size_hidden3 = 16
size_hidden4 = 32
size_hidden5 = 16
size_output = 2  
device = 'gpu'

# Define your headers
headers = ["Iteration", "Epoch", "Train Acc", "Val Acc", "Confusion Matrix"]

# Open the file in write mode to create it or clear it if it already exists
with open('finalRun.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

for x in range(10):

    # Ensure reproducibility
    np.random.seed(1234+x)
    tf.random.set_seed(1234+x)

    # Instantiate the model with the new configuration
    mlp_model = MLP(size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_hidden5, size_output, vocab_size, embedding_dim, device)


    # Training configuration
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.045)
    NUM_EPOCHS = 10

    # Training loop
    time_start = time.time()

    # Parameters
    batch_size = 2048  # You can adjust this size as needed

    # Convert the training and validation data into tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


    early_stopping_criteria_met = False

    for epoch in range(NUM_EPOCHS):
        train_accuracy.reset_states()
        val_accuracy.reset_states()
        
        total_train_loss = 0
        total_val_loss = 0
        train_batches = 0
        val_batches = 0

        for X_batch, y_batch in train_dataset:
            grads = mlp_model.backward(X_batch, y_batch)
            optimizer.apply_gradients(zip(grads, mlp_model.variables))
            
            train_predictions = mlp_model.forward(X_batch)
            batch_train_loss = mlp_model.loss(train_predictions, y_batch).numpy()
            
            total_train_loss += batch_train_loss
            train_batches += 1
            train_accuracy.update_state(y_batch, train_predictions)
        
        for X_batch, y_batch in val_dataset:
            val_predictions = mlp_model.forward(X_batch)
            batch_val_loss = mlp_model.loss(val_predictions, y_batch).numpy()
            
            total_val_loss += batch_val_loss
            val_batches += 1
            val_accuracy.update_state(y_batch, val_predictions)
        
        avg_train_loss = total_train_loss / train_batches
        avg_val_loss = total_val_loss / val_batches
        train_acc = train_accuracy.result().numpy()
        val_acc = val_accuracy.result().numpy()

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for early stopping criteria
        if train_acc > 0.73 and abs(train_acc - val_acc) < 0.03:
            print("Early stopping criteria met")
            early_stopping_criteria_met = True
            break

    if not early_stopping_criteria_met:
        print("Early stopping criteria not met, completed all epochs")

    time_taken = time.time() - time_start
    print(f"Training completed in {time_taken:.2f} seconds.")

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    y_pred_test = []
    y_true_test = []

    for X_batch, y_batch in test_dataset:
        test_predictions = mlp_model.forward(X_batch)
        y_pred_test.extend(extract_predictions(test_predictions))
        y_true_test.extend(tf.argmax(y_batch, axis=1).numpy())

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    print('Confusion Matrix for iteration {}:'.format(x+1))
    print(cm)

    time_taken = time.time() - time_start
    print(f"Training completed in {time_taken:.2f} seconds.")

    with open('finalRun.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Appending the confusion matrix along with the accuracy data
        writer.writerow([x+1, "Final", train_acc, val_acc, cm.tolist()]) # cm is converted to list for CSV compatibility