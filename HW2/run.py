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


layerDepth = [16, 32, 64, 128, 256]
output_dim = [16, 32, 64]

# Calculate total iterations
total_iterations = len(output_dim) * len(layerDepth)**5  # 5 nested loops of layerDepth
current_iteration = 0


layerDepth = [16, 32, 64, 128, 256]
output_dim = [16, 32, 64]

# Generate all possible configurations
configurations = [
    {
        "config": (output_dim_chosen, l1, l2, l3, l4, l5),
        "iteration": i+1,  # Iteration numbers start at 1
        "trainingAccuracy": None,  # Initially empty
        "testAccuracy": None  # Initially empty
    }
    for i, (output_dim_chosen, l1, l2, l3, l4, l5) in enumerate(
        itertools.product(output_dim, layerDepth, layerDepth, layerDepth, layerDepth, layerDepth)
    )
]

# Set the starting iteration
start_iteration = 2486

# Iterate through configurations starting from the specified iteration
for config in configurations:  # Adjust for zero-based indexing
    # Your code to train and evaluate the model here
    # For example, replace the next two lines with your model training and evaluation
    training_accuracy = 0.0  # Placeholder for actual training accuracy
    test_accuracy = 0.0  # Placeholder for actual test accuracy
    
    # Update the configuration with actual results
    config["trainingAccuracy"] = training_accuracy
    config["testAccuracy"] = test_accuracy
    
    # Optional: print current progress or log it
    print(f"Processed {config['iteration']} / {len(configurations)}")


results_path = "model_configurations_with_results.json"
with open(results_path, 'w') as file:
    json.dump(configurations, file)

print(f"We have initialized all iterations into json memory {results_path}.")




if os.path.exists("run.json"):
    with open("run.json", 'r') as file:
        # Load the existing configurations from run.json
        saved_configurations = json.load(file)
        
        # Ensure the loaded configurations is not empty and is a list
        if saved_configurations and isinstance(saved_configurations, list):
            # Update your local configurations array
            configurations = saved_configurations

        print("We have updates json memory")
else:
    print("run.json not found. Starting from scratch.")


for config in configurations[start_iteration - 1:]:  # Adjust for zero-based indexing
    # Unpack the configuration directly, since it's already a tuple of integers
    outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth = config["config"]
    
    # Create and compile the model
    model = create_model(outputDimChosen, l1_depth, l2_depth, l3_depth, l4_depth, l5_depth)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_padded, y_train, epochs=5, batch_size=2048, validation_data=(X_test_padded, y_test), verbose=0, shuffle=True)
    
    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    
    # Update the configuration with actual results
    config["trainingAccuracy"] = history.history['accuracy'][-1] * 100  # Convert to percentage
    config["testAccuracy"] = accuracy * 100  # Convert to percentage
    
    # Open run.json file to append the result after each model evaluation
    with open("run.json", "w") as file:
        json.dump(configurations, file, indent=4)  # Use indent for better readability
    
    # Print current progress
    print(f"Processed {config['iteration']} / {len(configurations)} - Config: {config['config']} - Training Acc: {config['trainingAccuracy']}, Test Acc: {config['testAccuracy']}")

print("All configurations processed and results saved to run.json.")
