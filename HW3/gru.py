import tensorflow as tf
import numpy as np
import os
import time
## code adopted from tf, pytorch and karpathy blog

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# Take a look at the first 400 characters in text
print(text[:400])
# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')
example_texts = ['NLPUSF', 'Assignment3']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
chars = chars_from_ids(ids)
tf.strings.reduce_join(chars, axis=-1).numpy()

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))
seq_length = 140
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))

for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 256

class NLPUSFModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
    
model = NLPUSFModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", example_batch_mean_loss)
tf.exp(example_batch_mean_loss).numpy()

model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 20
# Start training your model
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
  
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['Queen:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

# ## Simple beam search pseudocode, adapt this to
# function BEAM_SEARCH(RNN, start_sequence, beam_width):
#     # RNN: the recurrent neural network model for sequence generation (custom LSTM, GRU, custom Elman RNN)
#     # start_sequence: the initial part of the sequence (could be just a start symbol or set of symbols)
#     # beam_width: the number of sequences to keep at each step -- This is another hyper-parameter, play with it, as discussed in class, beam search will still provide you sub-optimal solution

#     Initialize an empty list `candidates` to store current sequence candidates -- One can use other datastructures, to optimize overall workeflow
#     Initialize an empty list `final_candidates` to store completed sequences
    
#     Add start_sequence to `candidates` with its score (e.g., log likelihood)

#     while not all sequences in `candidates` are complete:
#         Initialize an empty list `all_expansions` for storing all possible next steps

#         for each sequence in `candidates`:
#             if the sequence is complete:
#                 Add it to `final_candidates`
#                 Continue to the next iteration

#             Predict the next step probabilities using RNN given the current sequence
#             Select top-k next steps (where k is the beam width) based on probabilities

#             for each next step in top-k:
#                 Create a new sequence by appending the next step to the current sequence
#                 Calculate the new sequence's score (e.g., update log likelihood)
#                 Add the new sequence and its score to `all_expansions`

#         Sort `all_expansions` by score in descending order
#         Keep only the top `beam_width` sequences in `all_expansions`
#         Replace `candidates` with `all_expansions`

#     Add any remaining sequences in `candidates` to `final_candidates`
#     Sort `final_candidates` by score in descending order

#     return the top sequence from `final_candidates` (or top-N sequences if desired)

# # Usage example
# 1. RNN = InitializeYourRNNModel()
# 2. start_sequence = ["<start>"]  # Example start symbol
# 3. beam_width = 5  # Example beam width
# 4. best_sequence = BEAM_SEARCH(RNN, start_sequence, beam_width)
# 5. print("Best sequence:", best_sequence)
# Check above step on one-step this will provide you with tricks that will be useful to create beam-search


# Things to do
# 1. Integrate custom_beamsearch with your models
# 1. Optimize your hyper-parameter --> Learning rate, hidden_size, layers, optimizer, epochs, batch_size
# 2. Divide dataset into train, validation, and test, once your model gets reasonable performance (lower loss), then test the story generation capability of your system
# 3. Replace GRU with custom LSTM shared with you and test how it works
# 4. Create custom Elman RNN (h_t = tanh(X_tW + Uh_{t-1} + b)) and compare performance across different RNNs (Custom_ElmanRNN, GRU, Custom_LSTM). Also provide loss curves for each models and saved weights.
# 5. Provide statistical significance of your model
# 6. Show different texts generated by your models