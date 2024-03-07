import nltk
from nltk.corpus import brown
from nltk.util import trigrams, bigrams
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from datetime import datetime
import csv

# Configuration variable to run all calculations in parallel
PARALLEL_PROCESS = True

# Alphas for MLE optimization
ALPHAS = [1, 0.1, 0.01]

def worker_function(chunk_of_trigrams, bigrams_list, alpha, unique_tokens, worker_id):
    print(f"Worker {worker_id} starting with alpha={alpha}")
    probabilities = {}
    N = len(unique_tokens)
    
    # Ensure that chunk_of_trigrams is a list of tuples
    chunk_of_trigrams_list = [tuple(trigram) for trigram in chunk_of_trigrams.tolist()]
    
    for i, trigram in enumerate(chunk_of_trigrams_list):
        if (i + 1) % (len(chunk_of_trigrams_list) // 10) == 0:
            percent_complete = (i + 1) / len(chunk_of_trigrams_list) * 100
            print(f"Worker {worker_id}: Approximately {percent_complete:.2f}% of trigrams processed for alpha={alpha}")
        
        x_y = trigram[:2]
        count_xyz = chunk_of_trigrams_list.count(trigram)
        count_xy = bigrams_list.count(x_y)
        p_smooth = (count_xyz + alpha) / (count_xy + alpha * N)
        probabilities[trigram] = p_smooth
    
    print(f"Worker {worker_id} finished with alpha={alpha}")
    return probabilities

def split_into_chunks(data, num_chunks):
    """
    Splits data into a specified number of chunks for parallel processing.
    
    :param data: Data to split.
    :param num_chunks: Number of chunks to split the data into.
    :return: List of data chunks.
    """
    return np.array_split(data, num_chunks)

def combine_results(results):
    """
    Combines results from all worker functions after parallel computation.
    
    :param results: List of dictionaries containing probabilities from each worker.
    :return: Combined dictionary with all probabilities.
    """
    combined = {}
    for result in results:
        combined.update(result)
    return combined

def parallel_conditional_probability(trigrams_list, bigrams_list, alpha):
    """
    Manages the parallel computation of trigram probabilities using all available CPU cores.
    
    :param trigrams_list: List of all trigrams from the training set.
    :param bigrams_list: List of all bigrams from the training set.
    :param alpha: Smoothing parameter for probability calculation.
    :return: Dictionary of combined results with trigram probabilities.
    """
    cpu_count = os.cpu_count()  # Use all available CPU cores
    num_chunks = cpu_count  # One chunk per core
    chunks = split_into_chunks(trigrams_list, num_chunks)

    unique_tokens = set([token for trigram in trigrams_list for token in trigram])

    # Create a pool of processes and map the worker function to the chunks
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(worker_function, chunk, bigrams_list, alpha, unique_tokens, idx) 
                   for idx, chunk in enumerate(chunks)]
        results = []
        for future in futures:
            result = future.result()  # Blocks until the future is done
            results.append(result)
            print(f"Chunk processed by worker with result size: {len(result)}")  # Log progress
    
    return combine_results(results)

def main():
    """
    Main function to orchestrate the process of loading the Brown corpus,
    splitting into training and test sets, calculating trigram probabilities,
    and saving the results into separate CSV files for different alpha values.
    """
    # Ensure required NLTK datasets are downloaded
    nltk.download('brown')
    nltk.download('punkt')
    
    # Configuration for Group C
    x = 3  # Group C
    n = 10000
    start_index = (x - 1) * n
    end_index = x * n - 1

    # Extract sentences for Group C
    group_sentences = brown.sents()[start_index:end_index + 1]

    # Split the Group C sentences into training and test sets
    split_index = int(len(group_sentences) * 0.8)
    training_sentences = group_sentences[:split_index]
    test_sentences = group_sentences[split_index:]

    # Save the training and test sentences to CSV files
    with open('training_sentences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for sentence in training_sentences:
            writer.writerow([' '.join(sentence)])

    with open('test_sentences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for sentence in test_sentences:
            writer.writerow([' '.join(sentence)])

    # Generate trigrams and bigrams from the training sentences
    all_trigrams_flat = [trigram for sentence in training_sentences for trigram in trigrams(sentence)]
    all_bigrams = [bigram for sentence in training_sentences for bigram in bigrams(sentence)]

    for alpha in ALPHAS:
        print(f"Calculating probabilities in parallel for alpha={alpha}...")
        conditional_probabilities = parallel_conditional_probability(all_trigrams_flat, all_bigrams, alpha)
        
        csv_file_path = f"trigram_probabilities_{alpha}.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trigram', 'Probability'])
            for trigram, probability in conditional_probabilities.items():
                trigram_str = ' '.join(trigram)
                writer.writerow([trigram_str, probability])
        
        print(f"Probabilities for alpha={alpha} saved to {csv_file_path}.")

if __name__ == '__main__':
    main()