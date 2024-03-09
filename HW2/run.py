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


def create_model(outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth):
    model = Sequential([
        Embedding(input_dim=10000, output_dim=outputDimChosen, input_length=45),
        Flatten(),
        Dense(l1_depth),
        Dense(l2_depth),
        Dense(l3_depth),
        Dense(l4_depth),
        Dense(l5_depth),
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



def train_and_evaluate_model(params):
    outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth = params
    model = create_model(outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_padded, y_train, epochs=10, batch_size=2048, validation_data=(X_test_padded, y_test), verbose=0)
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    return accuracy * 100  # Return test accuracy

layerDepth = [16, 32, 64, 128, 256]
output_dim = [16, 32, 64]

# Calculate total iterations
total_iterations = len(output_dim) * len(layerDepth)**5  # 5 nested loops of layerDepth
current_iteration = 0

# Iterate through every possible combination of output dimension and layer depths
for outputDimChosen in output_dim:
    for l1_depth in layerDepth:
        for l2_depth in layerDepth:
            for l3_depth in layerDepth:
                for l4_depth in layerDepth:
                    for l5_depth in layerDepth:
                        # Increment the current iteration
                        current_iteration += 1

                        # Calculate iterations left
                        iterations_left = total_iterations - current_iteration

                        # Print the current configuration and the progress
                        print(f"Config: {outputDimChosen}, {l1_depth}, {l2_depth}, {l3_depth}, {l4_depth}, {l5_depth}")
                        print(f"Iteration: {current_iteration}/{total_iterations} - Iterations left: {iterations_left}")

                        # Create and compile the model with updated regularization strengths
                        model = create_model(outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth)  # Assuming 'create_model()' uses the updated regularization strengths
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                        # Train the model
                        history = model.fit(X_train_padded, y_train, epochs=10, batch_size=2048, validation_data=(X_test_padded, y_test),verbose=0)

                        # Evaluate the model on test data
                        loss, accuracy = model.evaluate(X_test_padded, y_test,verbose=0)

                        # Extract the last training accuracy
                        training_accuracy = history.history['accuracy'][-1] * 100  # Convert to percentage

                        # Check conditions (assuming the conditions mentioned before)
                        test_accuracy_threshold = 78.6
                        difference_threshold = 10.0
                        actual_difference = abs(training_accuracy - accuracy * 100)

                        print("Training_accuracy:"+str(training_accuracy))
                        print("Testaccuracy:"+str(accuracy*100))

                        if accuracy * 100 > test_accuracy_threshold and actual_difference < difference_threshold:
                            # Save the model with a timestamp
                            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                            model_path = f"model_{timestamp}.h5"
                            model.save(model_path)
                            print(f"Model saved as {model_path} because it met the criteria.")

                            # Save the regularization weights to a text or JSON file
                            regularization_weights = {
                                'training_accuracy':training_accuracy,
                                'test_accuracy':accuracy,
                                'outputDimChosen':outputDimChosen, 
                                'l1_depth':l1_depth, 
                                'l2_depth':l2_depth, 
                                'l3_depth':l3_depth, 
                                'l4_depth':l4_depth, 
                                'l5_depth':l5_depth
                            }
                            regularization_path = f"regularization_weights_{timestamp}.json"
                            with open(regularization_path, 'w') as f:
                                json.dump(regularization_weights, f)
                            print(f"Regularization weights saved as {regularization_path}.")