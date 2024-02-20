import nltk
from nltk.corpus import brown
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Simple feature extraction: Bag of Words model
def extract_features(sentences):
    print("Extracting features...")
    vocabulary = Counter([word.lower() for sentence in sentences for word in sentence])
    vocab_list = list(vocabulary.keys())
    features = np.zeros((len(sentences), len(vocab_list)))
    for i, sentence in enumerate(sentences):
        sentence_counts = Counter([word.lower() for word in sentence])
        for word, count in sentence_counts.items():
            if word in vocab_list:
                features[i, vocab_list.index(word)] = count
    print("Features extracted.")
    return features, vocab_list

# Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Get k nearest neighbors
def get_neighbors(features, test_instance, k):
    distances = np.array([euclidean_distance(test_instance, instance) for instance in features])
    neighbors = distances.argsort()[:k]
    return neighbors

# Predict the class based on nearest neighbors
def predict_classification(train_labels, neighbors):
    votes = Counter(train_labels[neighbor] for neighbor in neighbors)
    predicted_label = votes.most_common(1)[0][0]
    return predicted_label

# Parallel KNN prediction with progress tracking
def knn_predict_parallel(train_features, train_labels, test_features, k, max_workers):
    print("Starting parallel KNN prediction...")
    total = len(test_features)
    completed = 0
    
    def process_instance(test_instance):
        neighbors = get_neighbors(train_features, test_instance, k)
        return predict_classification(train_labels, neighbors)
    
    predictions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {executor.submit(process_instance, instance): i for i, instance in enumerate(test_features)}
        for future in as_completed(future_to_instance):
            predictions.append(future.result())
            completed += 1
            print(f"Completed {completed}/{total} predictions.")
    print("Parallel KNN prediction completed.")
    return predictions

# Evaluate metrics
def evaluate_metrics(true_labels, predicted_labels):
    print("Evaluating metrics...")
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))
    
    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Metrics evaluated.")
    return accuracy, precision, recall, f1_score

# Main execution
if __name__ == "__main__":
    # Download necessary NLTK datasets
    nltk.download('brown')
    nltk.download('punkt')

    # Configuration for a specific group of sentences
    x = 3  # Example group C
    n = 10000
    start_index = (x - 1) * n
    end_index = x * n - 1

    # Extract sentences from the Brown corpus
    group_sentences = brown.sents()[start_index:end_index + 1]

    # Feature extraction
    features, vocab_list = extract_features(group_sentences)
    split_point = int(len(features) * 0.8)
    train_features, test_features = features[:split_point], features[split_point:]
    train_labels, test_labels = np.random.randint(2, size=split_point), np.random.randint(2, size=len(features) - split_point)

    k = 3
    num_of_workers = 4

    # Predict labels for the test set using parallel processing
    predictions = knn_predict_parallel(train_features, train_labels, test_features, k,num_of_workers)

    # Evaluation metrics
    accuracy, precision, recall, f1_score = evaluate_metrics(test_labels, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")