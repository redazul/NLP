import pandas as pd
import nltk
from collections import Counter

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
import tensorflow as tf

nltk.download('punkt')

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
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.4, random_state=42)

# Further split for validation set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Define the model
model = Sequential([
    Embedding(input_dim=unique_vocab + 1, output_dim=100, input_length=45),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Training the model
epochs = 10  # Number of epochs to train for
batch_size = 32  # Batch size for training

# Fit the model on the training data
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=2)

# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")