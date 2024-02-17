import csv
import numpy as np
from collections import Counter

def calculate_perplexity(trigram_probabilities, test_sentences):
    total_log_probability = 0
    N = 0  # Total number of trigrams
    
    for sentence in test_sentences:
        sentence = ['<s>', '<s>'] + sentence + ['</s>']  # Assuming <s> and </s> are start and end tokens
        trigrams = [tuple(sentence[i:i+3]) for i in range(len(sentence)-2)]
        for trigram in trigrams:
            if trigram in trigram_probabilities:
                total_log_probability += np.log(trigram_probabilities[trigram])
            else:
                total_log_probability += np.log(1e-12)  # Small probability for unknown trigrams
        N += len(trigrams)
    
    perplexity = np.exp(-total_log_probability / N)
    return perplexity

# Load trigram probabilities from the CSV file
def load_trigram_probabilities(csv_file_path):
    probabilities = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            trigram_str, probability = row
            trigram = tuple(trigram_str.split())
            probabilities[trigram] = float(probability)
    return probabilities


# Function to calculate the likelihood of the data given a hypothesis (language model) using pre-calculated probabilities
def calculate_likelihood_from_csv(trigram_probabilities, n_gram, alpha, vocabulary_size):
    """
    Calculate the likelihood P(D|H) using pre-calculated probabilities.
    :param trigram_probabilities: Dictionary of trigram probabilities.
    :param n_gram: The n-gram for which to calculate the likelihood.
    :param alpha: Smoothing parameter.
    :param vocabulary_size: The number of unique words in the vocabulary.
    :return: Likelihood P(D|H).
    """
    # Fall back to a smoothed probability if the trigram is not found
    if n_gram in trigram_probabilities:
        return trigram_probabilities[n_gram]
    else:
        return alpha / (alpha * vocabulary_size)  # Simplified smoothing for unknown trigrams

# Function to calculate the prior probability of a hypothesis
# This function remains unchanged
def calculate_prior(n_gram_counts, alpha, vocabulary_size):
    """
    Calculate the prior probability P(H) of a hypothesis (n-gram occurrence).
    """
    total_possible_n_grams = len(n_gram_counts) + alpha * vocabulary_size
    return 1 / total_possible_n_grams

# Since the evidence or marginal likelihood P(D) might depend on the entire dataset's probabilities,
# and you're using pre-calculated probabilities, calculating evidence accurately might not be straightforward.
# You might need to adjust this calculation based on your specific requirements or assumptions about the distribution of probabilities.

# For this example, let's proceed with calculating the posterior probability using pre-calculated likelihoods and a simplified approach for prior and evidence.

# Assuming some setup variables are defined, such as vocabulary_size and alpha

ALPHAS = [1, 0.1, 0.01]

for alpha in ALPHAS:

    csv_file_path = f'trigram_probabilities_{alpha}.csv'
    trigram_probabilities = load_trigram_probabilities(csv_file_path)

    vocabulary_size = len(set(trigram_probabilities.keys()))


    # Now, let's assume you have observed a trigram and want to calculate its posterior probability
    observed_trigram = ('few', 'newer', 'homes')  # This is the data D

    # Calculate likelihood using pre-calculated probabilities
    likelihood = calculate_likelihood_from_csv(trigram_probabilities, observed_trigram, alpha, vocabulary_size)

    # For demonstration purposes, let's simplify prior and evidence calculations
    prior = calculate_prior(Counter(trigram_probabilities.keys()), alpha, vocabulary_size)
    # Simplified evidence calculation as an example
    evidence = 1  # This is a placeholder; in practice, calculate this based on your model's needs

    # Calculate the posterior probability using Bayes' Theorem
    posterior = likelihood * prior / evidence

    print(f"Posterior probability of the trigram {observed_trigram}: {posterior}")


    # Assuming 'sentences' is a list of tokenized sentences for testing
    # For example, this could be a small subset of sentences from your corpus
    test_sentences = [['few', 'newer', 'homes']]

    

    # Now call calculate_perplexity correctly
    perplexity = calculate_perplexity(trigram_probabilities, test_sentences)
    print(f"Perplexity of the model with alpha({alpha}) : {perplexity}")
