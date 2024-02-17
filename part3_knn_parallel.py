import numpy as np
import csv
import nltk
from nltk.tokenize import word_tokenize
import time
import concurrent.futures
import os

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
            if trigram in trigram_index_dict:
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

def calculate_distance(test_point, train_points):
    # Efficiently calculate distances using NumPy broadcasting
    return np.sqrt(np.sum((train_points - test_point) ** 2, axis=1))

def calculate_distances_in_parallel(test_point, train_features):
    # Function to split the training data into chunks and calculate distances in parallel
    num_workers = os.cpu_count()  # Number of parallel workers
    chunk_size = int(np.ceil(len(train_features) / num_workers))
    chunks = [train_features[i:i + chunk_size] for i in range(0, len(train_features), chunk_size)]
    
    distances = np.array([])
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(calculate_distance, test_point, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            distances = np.concatenate((distances, future.result()))
    return distances

def knn_classifier_parallel_modified(args):
    test_point, train_features, k, index = args
    print(f"Processing test point {index+1} with parallel distance calculations")
    distances = calculate_distances_in_parallel(test_point, train_features)
    k_nearest_neighbors = np.argsort(distances)[:k]
    return k_nearest_neighbors

def parallel_knn_classification(test_features, train_features, k, workers):
    args = [(test_point, train_features, k, i) for i, test_point in enumerate(test_features)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_knn = {executor.submit(knn_classifier_parallel_modified, arg): arg for arg in args}
        for future in concurrent.futures.as_completed(future_to_knn):
            results.extend(future.result())
    return results

if __name__ == "__main__":
    start_time = time.time()

    train_features, trigram_list = load_and_preprocess_data('trigram_probabilities_0.01.csv')
    test_features = preprocess_data_test_sentences('test_sentences.csv', trigram_list)
    k = 5

    print(f"Total number of test points to process: {len(test_features)}")
    results = parallel_knn_classification(test_features, train_features, k, os.cpu_count())

    unique_labels = np.unique(results)
    num_clusters = len(unique_labels)
    
    end_time = time.time()
    print(f"\nNumber of clusters/classifications made: {num_clusters}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
