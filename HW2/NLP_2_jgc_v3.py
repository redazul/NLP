import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import time

# Load data
final_processed_sentences = pd.read_csv('final_processed_sentences.csv', header=None)
sentiments = pd.read_csv('sentiments.csv', header=None)

# Convert sentiments to binary labels
sentiments = sentiments[0].map({'positive': 1, 'negative': 0}).values

# Initialize tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(final_processed_sentences[0])
vocab_size = len(tokenizer.word_index) + 1  # Plus one for padding token

# Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(final_processed_sentences[0])

# Pad sequences to ensure uniform length
max_length = 45
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, sentiments, test_size=0.3, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)

# Preprocess labels for the model
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

np.random.seed(1234)
tf.random.set_seed(1234)

size_hidden1 = 128
size_hidden2 = 128
size_hidden3 = 128
size_output = 10

# Define class to build mlp model
## Change this class to add more layers
class MLP(object):
 def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_output, device=None):
    """
    size_input: int, size of input layer
    size_hidden1: int, size of the 1st hidden layer
    size_hidden2: int, size of the 2nd hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden1, self.size_hidden2, self.size_hidden3, self.size_output, self.device =\
    size_input, size_hidden1, size_hidden2, size_hidden3, size_output, device
    
    # Initialize weights between input mapping and a layer g(f(x)) = layer
    self.W1 = tf.Variable(tf.random.normal([self.size_input, self.size_hidden1],stddev=0.1)) # Xavier(Fan-in fan-out) and Orthogonal
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.zeros([1, self.size_hidden1])) # 0 or constant(0.01)
    
    # Initialize weights between input layer and 1st hidden layer
    self.W2 = tf.Variable(tf.random.normal([self.size_hidden1, self.size_hidden2],stddev=0.1))
    # Initialize biases for hidden layer
    self.b2 = tf.Variable(tf.zeros([1, self.size_hidden2]))
    
    
    # Assuming the last layer of your model is something like this:
    self.W3 = tf.Variable(tf.random.normal([self.size_hidden2, size_output], stddev=0.1))
    self.b3 = tf.Variable(tf.zeros([1, size_output]))

    
    # Define variables to be updated during backpropagation
    self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
  
 def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)
      
    return self.y

 def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    #y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_x = cce(y_true_tf, y_pred_tf)
    # Use keras or tf_softmax, both should work for any given model
    #loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf, labels=y_true_tf))
    
    return loss_x

 def backward(self, X_train, y_train):
    """
    backward pass
    """
    
    
    with tf.GradientTape() as tape:
        
      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)

        
    grads = tape.gradient(current_loss, self.variables)

    return grads
    
           
 def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    #X_tf = X
    
    # Compute values in hidden layers
    h1 = tf.matmul(X_tf, self.W1) + self.b1
    z1 = tf.nn.relu(h1)
    
    h2 = tf.matmul(z1, self.W2) + self.b2
    z2 = tf.nn.relu(h2)
    

    # Compute output
    output = tf.nn.softmax(tf.matmul(z2, self.W3) + self.b3)    
    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this 
    # Second add tf.Softmax(output) and then return this variable
    return (output)

# Set number of epochs
NUM_EPOCHS = 1000
batch_size = 1024  # You can adjust this based on your GPU's memory

size_output = 2  # For binary classification (positive/negative)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
# Initialize model using CPU
# Correct the size_input based on your padded sequences' shape
size_input = max_length  # This matches the input dimension to your model

# Initialize model specifying the input size and the number of classes
mlp_on_gpu = MLP(size_input, size_hidden1, size_hidden2, size_hidden3, size_output, device='gpu')

# Begin training
time_start = time.time()

train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()


# Convert training and validation data into TensorFlow datasets, then batch them
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

for epoch in range(NUM_EPOCHS):
    train_accuracy_metric.reset_states()
    val_accuracy_metric.reset_states()
    
    # Training loop - iterate over batches
    for x_batch_train, y_batch_train in train_dataset:
        y_pred_train = mlp_on_gpu.forward(x_batch_train)
        loss = mlp_on_gpu.loss(y_pred_train, y_batch_train)
        train_accuracy_metric.update_state(y_batch_train, y_pred_train)
        
        grads = mlp_on_gpu.backward(x_batch_train, y_batch_train)
        optimizer.apply_gradients(zip(grads, mlp_on_gpu.variables))
        
    # Validation loop - iterate over batches
    for x_batch_val, y_batch_val in val_dataset:
        y_pred_val = mlp_on_gpu.forward(x_batch_val)
        val_loss = mlp_on_gpu.loss(y_pred_val, y_batch_val)
        val_accuracy_metric.update_state(y_batch_val, y_pred_val)
        

    print(f"Epoch {epoch}, Loss: {loss.numpy()}, Accuracy: {train_accuracy_metric.result().numpy()}, Validation Loss: {val_loss.numpy()}, Validation Accuracy: {val_accuracy_metric.result().numpy()}")


time_taken = time.time() - time_start
print(f"Training completed in {time_taken:.2f} seconds.")

