import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout

# Load sentences from CSV
def load_sentences(csv_file):
    sentences = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                sentences.append(row[0])
    return sentences

# Load sentiments from CSV and convert to numeric values
def load_sentiments(csv_file):
    sentiments = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                # Convert 'positive' to 1 and 'negative' to 0
                sentiment_value = 1 if row[0].lower() == 'positive' else 0
                sentiments.append(sentiment_value)
    return sentiments


# Tokenize sentences and pad them
def tokenize_and_pad(sentences, max_length=45):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences, tokenizer.word_index

# Build the MLP model
def build_model(vocab_size, embedding_dim=45, max_length=45):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length,mask_zero=True),
        Flatten(),  # This is a simple model; for complex data, consider using RNN or Conv1D
        Dense(128, activation='relu'),  # First hidden layer
        Dropout(0.5),  # To prevent overfitting; adjust as necessary
        # Add more layers as needed
        Dense(1, activation='sigmoid')  # Output layer
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to process data and train model
def process_data_and_train_model(sentences_csv, sentiments_csv):
    sentences = load_sentences(sentences_csv)
    sentiments = load_sentiments(sentiments_csv)

    print("\n\nChecking how pad has been embedded")
    print(sentences[9])
    print(sentences[14])

    # Ensure the length of sentences and sentiments match
    assert len(sentences) == len(sentiments), "The lengths of sentences and sentiments do not match."

    padded_sequences, word_index = tokenize_and_pad(sentences)

    print("\n\nChecking how pad has been embedded")
    print(padded_sequences[9])
    print(padded_sequences[14])

    vocab_size = len(word_index) + 1
    model = build_model(vocab_size)

    # Convert sentiments to a numpy array for TensorFlow
    sentiments = np.array(sentiments)

    for x in range(len(padded_sequences)):
        padded_sequences[x]=abs(padded_sequences[x]-1)

    print("\nPost Masking Fix")
    print(padded_sequences[14])

    # Split data into training and validation sets
    split = int(len(sentences) * 0.8)
    X_train, X_val = padded_sequences[:split], padded_sequences[split:]
    y_train, y_val = sentiments[:split], sentiments[split:]

    # Train the model
    model.fit(X_train, y_train, epochs=70, validation_data=(X_val, y_val))

# Example usage
sentences_csv = 'final_processed_sentences.csv'
sentiments_csv = 'sentiments.csv'
process_data_and_train_model(sentences_csv, sentiments_csv)
