import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import L1L2


# Load your data
sentences = pd.read_csv('final_processed_sentences.csv', header=None)
sentiments = pd.read_csv('sentiments.csv', header=None)

# Convert labels to numeric if they're not already
sentiments[0] = pd.to_numeric(sentiments[0], errors='coerce')

# Calculate split indices
total_samples = sentences.shape[0]
train_end = int(total_samples * 0.6)
val_end = int(total_samples * 0.8)

# Manually split the datasets
X_train = sentences.iloc[:train_end, 0].astype(str)
y_train = sentiments.iloc[:train_end, 0].values.astype('float32')

X_val = sentences.iloc[train_end:val_end, 0].astype(str)
y_val = sentiments.iloc[train_end:val_end, 0].values.astype('float32')

X_test = sentences.iloc[val_end:, 0].astype(str)
y_test = sentiments.iloc[val_end:, 0].values.astype('float32')

# Preprocessing with Tokenizer and pad_sequences as before
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 45
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Model definition using TensorFlow 2.x API
model = Sequential([
    Embedding(vocab_size, 100, input_length=max_length),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(8, activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=512, validation_data=(X_val_pad, y_val), verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Saving the model
model.save('sentiment_analysis_model.h5')
