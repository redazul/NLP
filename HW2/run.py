import pandas as pd
import nltk
from collections import Counter
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
number_of_unique_words = len(set(combined_words_list))

print(number_of_unique_words)

if(number_of_unique_words!=10000):
    print("Vocab length is bad")
