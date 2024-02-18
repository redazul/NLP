import nltk
from nltk.corpus import brown
import numpy as np
from collections import Counter

# Simple feature extraction: Bag of Words model
def extract_features(sentences):
    vocabulary = Counter([word.lower() for sentence in sentences for word in sentence])
    vocab_list = list(vocabulary.keys())
    features = np.zeros((len(sentences), len(vocab_list)))
    for i, sentence in enumerate(sentences):
        sentence_counts = Counter([word.lower() for word in sentence])
        for word, count in sentence_counts.items():
            if word in vocab_list:
                features[i, vocab_list.index(word)] = count
    return features, vocab_list

# Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Get k nearest neighbors
def get_neighbors(features, test_instance, k):
    distances = np.array([euclidean_distance(test_instance, instance) for instance in features])
    neighbors = distances.argsort()[:k]
    print(f"Nearest neighbors' indices: {neighbors}")
    return neighbors

# Predict the class based on nearest neighbors
def predict_classification(train_labels, neighbors):
    votes = Counter(train_labels[neighbor] for neighbor in neighbors)
    predicted_label = votes.most_common(1)[0][0]
    print(f"Predicted label: {predicted_label}, Votes: {votes}")
    return predicted_label

# Main KNN function to tie everything together
def knn_predict(train_features, train_labels, test_features, k=3):
    predictions = []
    for i, test_instance in enumerate(test_features):
        print(f"Predicting for test instance {i+1}/{len(test_features)}...")
        neighbors = get_neighbors(train_features, test_instance, k)
        prediction = predict_classification(train_labels, neighbors)
        predictions.append(prediction)
    return predictions

def evaluate_metrics(true_labels, predicted_labels):
    # Convert to numpy arrays for easier manipulation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))
    
    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


# Download necessary NLTK datasets
nltk.download('brown')
nltk.download('punkt')

# Configuration for your specific group of sentences
x = 3  # Example group C
n = 10000
start_index = (x - 1) * n
end_index = x * n - 1

# Extract sentences from the Brown corpus
group_sentences = brown.sents()[start_index:end_index + 1]


print("\nPlease wait we're capturing train and test data\n")


# Example usage
features, vocab_list = extract_features(group_sentences)
# Example split for demonstration, in practice, you should use a proper train-test split
split_point = int(len(features) * 0.8)
train_features, test_features = features[:split_point], features[split_point:]
# Assuming binary classification for demonstration, you'll need to adjust based on your actual labels
train_labels, test_labels = np.random.randint(2, size=split_point), np.random.randint(2, size=len(features) - split_point)


# Predict labels for the test set
predictions = knn_predict(train_features, train_labels, test_features, k=3)

print("Predictions:", predictions)

# Calculate evaluation metrics
accuracy, precision, recall, f1_score = evaluate_metrics(test_labels, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")
