import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from collections import Counter


# Seed for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)


# Read final_processed_sentences.csv
final_processed_sentences = pd.read_csv('final_processed_sentences.csv', header=None)

# Read sentiments.csv
sentiments = pd.read_csv('sentiments.csv', header=None)

# Initialize empty lists to store sentences
positive_sentences_list = []
negative_sentences_list = []

# Iterate through indices of sentiments
for index, sentiment in sentiments.iterrows():
    if sentiment[0] == "positive":
        positive_sentences_list.append(final_processed_sentences.iloc[index, 0])
    elif sentiment[0] == "negative":
        negative_sentences_list.append(final_processed_sentences.iloc[index, 0])

# Tokenize each sentence and flatten the list of tokens for positive sentences
positive_tokens = [word for sentence in positive_sentences_list for word in nltk.word_tokenize(sentence)]

# Tokenize each sentence and flatten the list of tokens for negative sentences
negative_tokens = [word for sentence in negative_sentences_list for word in nltk.word_tokenize(sentence)]

# Count the frequency of each word for positive sentences
positive_word_frequency = Counter(positive_tokens)

# Count the frequency of each word for negative sentences
negative_word_frequency = Counter(negative_tokens)

# Extract just the words from sorted word frequency lists
sorted_positive_words = [word for word, _ in positive_word_frequency.most_common()]
sorted_negative_words = [word for word, _ in negative_word_frequency.most_common()]

# Find common words in positive and negative lists
common_words = set(sorted_positive_words).intersection(set(sorted_negative_words))

# Initialize dictionary to store difference in frequency
positive_word_frequency_difference = {}

# Calculate frequency difference for common words
for word in common_words:
    difference = positive_word_frequency[word] - negative_word_frequency[word]
    positive_word_frequency_difference[word] = difference

# Sort positive_word_frequency_difference by value (difference) in descending order
sorted_positive_word_frequency_difference = sorted(positive_word_frequency_difference.items(), key=lambda x: x[1], reverse=True)

# # Print the difference in frequency for each word (sorted)
# print("\nDifference in Frequency (Positive - Negative) - Sorted:")
# for word, diff in sorted_word_frequency_difference:
#     print(f"{word}: {diff}")


# Initialize dictionary to store difference in frequency
negative_word_frequency_difference = {}

# Calculate frequency difference for common words
for word in common_words:
    difference = negative_word_frequency[word] - positive_word_frequency[word]
    negative_word_frequency_difference[word] = difference

# Sort negative_word_frequency_difference by value (difference) in descending order
sorted_negative_word_frequency_difference = sorted(negative_word_frequency_difference.items(), key=lambda x: x[1], reverse=True)

# Create new lists containing the top 5,000 words from each sorted difference list
top_positive_words = [word for word, _ in sorted_positive_word_frequency_difference[:5000]]
top_negative_words = [word for word, _ in sorted_negative_word_frequency_difference[:5000]]

# Print the top 5,000 words from each sorted difference list
print("\nTop 5,000 Words in Positive Sentences:")
print(top_positive_words)

print("\nTop 5,000 Words in Negative Sentences:")
print(top_negative_words)

# Combine the lists
combined_words_list = top_positive_words + top_negative_words

# Find the number of unique words
unique_vocab = len(set(combined_words_list))

# Initialize the Tokenizer with the combined unique words
tokenizer = Tokenizer(num_words=unique_vocab + 1, oov_token="<OOV>")
tokenizer.fit_on_texts(combined_words_list)

# Convert the sentences to sequences
sequences = tokenizer.texts_to_sequences(final_processed_sentences[0])
padded_sequences = pad_sequences(sequences, maxlen=45, padding='post')

# Convert sentiment labels to binary format
labels = sentiments[0].apply(lambda x: 1 if x == "positive" else 0).values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.4)

# Further split for validation set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

class CustomMLP:
    def __init__(self, layers, reg_lambda):
        self.layers = layers
        self.reg_lambda = reg_lambda
        self.build()

    def build(self):
        self.weights = []
        self.biases = []
        for i, layer in enumerate(self.layers[:-1]):
            W = tf.Variable(tf.random.normal([self.layers[i], self.layers[i+1]], stddev=0.1))
            b = tf.Variable(tf.zeros([self.layers[i+1]]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        activation = tf.cast(X, dtype=tf.float32)
        for W, b in zip(self.weights, self.biases):
            z = tf.matmul(activation, W) + b
            activation = tf.nn.relu(z)
        # No activation on the last layer
        return z

    def compute_loss(self, logits, labels):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.weights]) * self.reg_lambda
        return cross_entropy_loss + l2_loss

    def compute_gradients(self, X, y):
        with tf.GradientTape() as tape:
            logits = self.forward(X)
            loss = self.compute_loss(logits, y)
        return tape.gradient(loss, self.weights + self.biases), loss

    def apply_gradients(self, gradients, optimizer):
        optimizer.apply_gradients(zip(gradients, self.weights + self.biases))

# Model and training configuration
model = CustomMLP(layers=[45, 128, 128, 128, 128, 2], reg_lambda=0.0000001)
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Training loop
NUM_EPOCHS = 100  # Example, adjust as needed
batch_size = 1024  # Example, adjust as needed

def calculate_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    true_labels = tf.argmax(labels, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), dtype=tf.float32))
    return accuracy.numpy()

# Adjusted training loop with accuracy calculation
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_train_accuracy = 0
    total_val_accuracy = 0

    # Training
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        gradients, batch_loss = model.compute_gradients(X_batch, y_batch)
        model.apply_gradients(gradients, optimizer)
        total_loss += batch_loss.numpy()

        # Calculate training accuracy in each batch
        train_logits = model.forward(X_batch)
        total_train_accuracy += calculate_accuracy(train_logits, y_batch)

    # Validation
    for i in range(0, X_val.shape[0], batch_size):
        X_batch_val = X_val[i:i+batch_size]
        y_batch_val = y_val[i:i+batch_size]

        # Calculate validation accuracy
        val_logits = model.forward(X_batch_val)
        total_val_accuracy += calculate_accuracy(val_logits, y_batch_val)

    # Averaging loss and accuracy
    avg_loss = total_loss / (X_train.shape[0] // batch_size)
    avg_train_accuracy = total_train_accuracy / (X_train.shape[0] // batch_size)
    avg_val_accuracy = total_val_accuracy / (X_val.shape[0] // batch_size)

    # Print metrics
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
