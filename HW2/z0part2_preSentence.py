import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize NLTK's lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove "br" tokens
    text = text.replace("br", "")
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords, single characters, and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    # Ensure the sentence is exactly 45 words - if someone likes or hates a movie....they usally tell your right away
    #Oh I love that movie here's why
    #Oh I hate that movie here's why
    if len(filtered_tokens) > 45:
        filtered_tokens = filtered_tokens[:45]  # Truncate to 45
    elif len(filtered_tokens) < 45:
        filtered_tokens += ['<pad>'] * (45 - len(filtered_tokens))  # Pad to 45
    # Return the tokens for further processing
    return filtered_tokens

# Read sentences from the CSV file and preprocess them
processed_sentences_tokens = []  # This will hold the tokenized processed sentences
sentence_count = 0
with open('sentences.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        if not row:
            continue
        sentence = row[0]
        processed_tokens = preprocess_text(sentence)
        processed_sentences_tokens.append(processed_tokens)
        sentence_count += 1
        if sentence_count % 100 == 0:
            print(f"Processed {sentence_count} sentences...")

# Convert token lists back to sentences
padded_processed_sentences = [" ".join(tokens) for tokens in processed_sentences_tokens]

# Save the padded and processed sentences to a new CSV file
processed_file_path = 'final_processed_sentences.csv'
with open(processed_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for sentence in padded_processed_sentences:
        writer.writerow([sentence])  # Write each padded and processed sentence as a row

print(f"Final processed sentences saved to {processed_file_path}. Total processed sentences: {sentence_count}")
