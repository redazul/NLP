import numpy as np
import csv
import nltk
from nltk.tokenize import word_tokenize
import time

# Download NLTK resources (required only once)
nltk.download('punkt')

def load_and_preprocess_data(csv_file_path):
    print("Starting to load and preprocess data...")
    start_time = time.time()
    trigram_index_dict = {}
    index = 0
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            trigram, probability = tuple(row[0].split()), float(row[1])
            if trigram not in trigram_index_dict:
                trigram_index_dict[trigram] = index
                index += 1
    
    # Initialize a feature vector for the entire dataset with zeros
    num_trigrams = len(trigram_index_dict)
    feature_vector = np.zeros(num_trigrams)
    
    # Reopen the file for reading the data again
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row again for actual loading
        for row in reader:
            trigram, probability = tuple(row[0].split()), float(row[1])
            if trigram in trigram_index_dict:  # Check if this step is necessary based on your logic
                trigram_index = trigram_index_dict[trigram]
                feature_vector[trigram_index] = probability
                
    end_time = time.time()
    print(f"Data loaded and preprocessed in {end_time - start_time:.2f} seconds.")
    return feature_vector, trigram_index_dict

def preprocess_data_test_sentences(csv_file_path, trigram_index_dict):
    print("Starting to preprocess test sentences...")
    start_time = time.time()
    test_features = []
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and not row[0].startswith('('):
                sentence = row[0]
                tokenized_sentence = word_tokenize(sentence.lower())
                feature_vector = np.zeros(len(trigram_index_dict))
                for i in range(len(tokenized_sentence) - 2):
                    trigram = (tokenized_sentence[i], tokenized_sentence[i+1], tokenized_sentence[i+2])
                    if trigram in trigram_index_dict:
                        trigram_index = trigram_index_dict[trigram]
                        feature_vector[trigram_index] = 1
                test_features.append(feature_vector)
    end_time = time.time()
    print(f"Test sentences preprocessed in {end_time - start_time:.2f} seconds.")
    return np.array(test_features)

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_classifier(args):
    test_point, train_features, k, index = args
    total_comparisons = len(train_features)  # Get the total number of comparisons
    distances = []
    for i, train_point in enumerate(train_features):
        # Log progress every 100 iterations with total comparisons, using carriage return to keep it on one line
        if i % 100 == 0:
            print(f"\rCalculating distance for test point {index+1}, comparison {i+1}/{total_comparisons}", end="")
        distances.append(calculate_distance(test_point, train_point))
    print()  # Print a newline to ensure the next output is on a new line
    k_nearest_neighbors = np.argsort(distances)[:k]
    return k_nearest_neighbors

if __name__ == "__main__":
    start_time = time.time()

    train_features, trigram_list = load_and_preprocess_data('trigram_probabilities_0.01.csv')
    test_features = preprocess_data_test_sentences('test_sentences.csv', trigram_list)
    k = 5
    total_test_points = len(test_features)  # Get the total number of test points
    print(f"Total number of test points to process: {total_test_points}")
    
    results = []

    for i, test_point in enumerate(test_features):
        print(f"Starting classification for test point {i+1}/{total_test_points}")
        start_point_time = time.time()
        k_nearest_neighbors = knn_classifier((test_point, train_features, k, i))
        results.extend(k_nearest_neighbors)
        end_point_time = time.time()
        print(f"Classification completed for test point {i+1}/{total_test_points} in {end_point_time - start_point_time:.2f} seconds.")

    unique_labels = np.unique(results)
    num_clusters = len(unique_labels)
    
    end_time = time.time()
    print(f"\nNumber of clusters/classifications made: {num_clusters}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    start_time = time.time()

    train_features, trigram_list = load_and_preprocess_data('trigram_probabilities_0.01.csv')
    test_features = preprocess_data_test_sentences('test_sentences.csv', trigram_list)
    k = 5
    results = []

    for i, test_point in enumerate(test_features):
        print(f"Starting classification for test point {i+1}/{len(test_features)}")
        start_point_time = time.time()
        k_nearest_neighbors = knn_classifier((test_point, train_features, k, i))
        results.extend(k_nearest_neighbors)
        end_point_time = time.time()
        print(f"Classification completed for test point {i+1}/{len(test_features)} in {end_point_time - start_point_time:.2f} seconds.")

    unique_labels = np.unique(results)
    num_clusters = len(unique_labels)
    
    end_time = time.time()
    print(f"\nNumber of clusters/classifications made: {num_clusters}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
