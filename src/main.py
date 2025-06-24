import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        # Track losses for plotting
        self.losses = []

    def sigmoid(self, x):
        """
        Sigmoid activation function - squashes any number to between 0 and 1
        
        Think of it like a dimmer switch:
        - Very negative numbers → close to 0 (off)
        - Very positive numbers → close to 1 (fully on)
        - Zero → 0.5 (halfway)
        """

        # Clip values to avoid overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivation(self, x):
        """
        How much the sigmoid function changes at point x
        This tells us how to adjust our weights during learning
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward pass - data flows from input to output
        Like following a recipe step by step
        """

        # Step 1: input layer to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1) # Apply the activation function

        # Step 2: Hidden layer to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2) # Apply the activation function

        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward pass - figure out how wrong we were and adjust
        Like tasting your cooking and adjusting the seasoning
        """

        # Get the number of training examples we supplied to the training method
        m = X.shape[0]

        # Calculate the error
        # Take the predicted iutput and subtract the expected output (y)
        # This tells how much of a difference there is between what was predicted and what we wanted
        output_error = output - y

        # How much should we adjust the output layer?
        output_delta = output_error * self.sigmoid_derivation(output)

        # How much should we change the hidden layer?
        # This is where the "chain rule" from calculus comes in
        # Think: if output is wrong, how much is each hidden neuron to blame?
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivation(self.a1)

        # Update weights and biases (the learning part!)
        # Learning rate controls how big steps we take
        # learning_rate = 0.1
        learning_rate = 1.0

        self.W2 -= self.a1.T.dot(output_delta) * learning_rate / m
        self.b2 -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate / m
        self.W1 -= X.T.dot(hidden_delta) * learning_rate / m
        self.b1 -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate / m

    def train(self, X, y, epochs):
        """
        Train the network by showing it examples many times
        Like practicing piano - repetition makes you better!
        """

        for epoch in range(epochs):
            # Perform a forward pass
            output = self.forward(X)

            # Calculate the loss (how wrong we were)
            loss = np.mean((output - y) ** 2) # Mean Squared Error
            self.losses.append(loss)

            # Perform a backward pass to adjust weights and biases
            self.backward(X, y, output)

            # Show progress every 100 epochs
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)
    
    def plot_loss(self):
        """
        Plot how the loss changes during training
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


# Example 1: Learn the XOR function
# XOR is like "either A or B, but not both"
print("=== Example 1: Learning XOR ===")
print("XOR truth table:")
print("0 XOR 0 = 0")
print("0 XOR 1 = 1") 
print("1 XOR 0 = 1")
print("1 XOR 1 = 0")
print()

# Create training data for XOR
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# Create the neural net and train it
nn_xor = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn_xor.train(X_xor, y_xor, epochs=50000)

# Test the trained network
print("Testing XOR predictions:")

predictions = nn_xor.predict(X_xor)
for i in range(len(X_xor)):
    input_val = X_xor[i]
    expected = y_xor[i][0]
    predicted = predictions[i][0]
    print(f"Input: {input_val}, Expected: {expected}, Predicted: {predicted:.3f}")

nn_xor.plot_loss()