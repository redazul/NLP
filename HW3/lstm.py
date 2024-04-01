import tensorflow as tf
from tensorflow import keras
# Here we provide a way to create CUSTOM LSTM and how to integrate this with Keras Workflow
class CustomLSTMCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]  # Hidden state size and cell state size

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # One can play with init to stabalize learning, remember what we discussed for MLP
        # As described in class LSTM is simply 4 different RNNs (h_t = sigma(Wx_t + Uh_{t-1} + b)) working in parallel, but connected jointly.
        # Weights for the input gate
        self.W_i = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_i')
        self.U_i = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_i')
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')

        # Weights for the forget gate
        self.W_f = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_f')
        self.U_f = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_f')
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f')

        # Weights for the cell state
        self.W_c = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_c')
        self.U_c = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_c')
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')

        # Weights for the output gate
        self.W_o = self.add_weight(shape=(input_dim, self.units), initializer='random_normal', name='W_o')
        self.U_o = self.add_weight(shape=(self.units, self.units), initializer='random_normal', name='U_o')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')

        super(CustomLSTMCell, self).build(input_shape)

    def call(self, inputs, states):
        h_tm1, c_tm1 = states  # Previous state

        # Input gate
        i = tf.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h_tm1, self.U_i) + self.b_i)

        # Forget gate
        f = tf.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(h_tm1, self.U_f) + self.b_f)

        # Cell state
        c_ = tf.tanh(tf.matmul(inputs, self.W_c) + tf.matmul(h_tm1, self.U_c) + self.b_c)
        c = f * c_tm1 + i * c_

        # Output gate
        o = tf.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(h_tm1, self.U_o) + self.b_o)

        # New hidden state
        h = o * tf.tanh(c)

        return h, [h, c]
    
units = 10  # Number of LSTM units
input_shape = (None, 5)  # Example input shape (timesteps, features)

# Create the LSTM layer using the custom cell
lstm_layer = keras.layers.RNN(CustomLSTMCell(units), input_shape=input_shape)

# Create a model using this layer
model = keras.Sequential([
    lstm_layer,
    keras.layers.Dense(1)  # Example output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
