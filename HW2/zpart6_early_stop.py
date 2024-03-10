import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
import datetime
import json
import itertools
import os
import time
from tensorflow.keras.callbacks import Callback





# Read final_processed_sentences.csv
final_processed_sentences = pd.read_csv('final_processed_sentences.csv', header=None)

# Read sentiments.csv
sentiments = pd.read_csv('sentiments.csv', header=None)

# Preprocess the data
X = final_processed_sentences[0].values
y = sentiments[0].apply(lambda x: 1 if x == 'positive' else 0).values  # Convert sentiment strings to binary labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vocab_size = 10000

# Tokenize the text data
tokenizer = Tokenizer(num_words=vocab_size)  # Vocabulary size
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_seq, maxlen=45, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=45, padding='post', truncating='post')

class CustomEarlyStopping(Callback):
    def __init__(self, min_accuracy=0.8, max_accuracy_diff=0.03):
        super(CustomEarlyStopping, self).__init__()
        # Minimum accuracy to meet the stopping criteria
        self.min_accuracy = min_accuracy
        # Maximum difference between training and validation accuracies
        self.max_accuracy_diff = max_accuracy_diff

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the training and validation accuracies for the current epoch
        train_accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        
        # Check if both accuracies are above the minimum threshold
        if train_accuracy > self.min_accuracy and val_accuracy > self.min_accuracy:
            # Check if the difference between the accuracies is within the specified limit
            if abs(train_accuracy - val_accuracy) < self.max_accuracy_diff:
                print(f"\nStopping training because the criteria have been met at epoch {epoch+1}")
                self.model.stop_training = True


def create_model(outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth, regularization_strength):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=outputDimChosen, input_length=45),
        Flatten(),
        Dense(l1_depth, activation='relu', kernel_regularizer=l2(regularization_strength)),
        Dense(l2_depth, activation='relu', kernel_regularizer=l2(regularization_strength)),
        Dense(l3_depth, activation='relu', kernel_regularizer=l2(regularization_strength)),
        Dense(l4_depth, activation='relu', kernel_regularizer=l2(regularization_strength)),
        Dense(l5_depth, activation='relu', kernel_regularizer=l2(regularization_strength)),
        Dense(1, activation='sigmoid')
    ])

    return model


# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# List available GPUs
print("Available GPU devices:", tf.config.list_physical_devices('GPU'))

# Verify CUDA GPU is available
if tf.test.is_built_with_cuda():
    print("TensorFlow was built with CUDA support")
else:
    print("TensorFlow was NOT built with CUDA support")


outputDimChosen = 32
l1_depth = 32
l2_depth = 16
l3_depth = 16
l4_depth = 32
l5_depth = 16


start_time = time.time()  # Record the start time of the process
times = []  # List to store the time taken for each iteration


iteration_start_time = time.time()  # Start time of the current iteration

regularization_strength=0.025

model = create_model(outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth, regularization_strength)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
custom_early_stopping = CustomEarlyStopping()

# Now, include this callback in your model.fit() method along with any other callbacks you might have
history = model.fit(X_train_padded, y_train, epochs=70, batch_size=2048, validation_data=(X_test_padded, y_test), shuffle=True, callbacks=[custom_early_stopping])


# Evaluate the model on test data
loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)

# Extract the training accuracy from the last epoch
train_accuracy = history.history['accuracy'][-1]  # Get the last training accuracy

# Print accuracies for the current iteration
print(f"Regularization Strength: {regularization_strength:.5f}, Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")