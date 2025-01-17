---

# Quantum Variational Autoencoder (VAE) with MNIST

This project demonstrates how to implement and train a Quantum Variational Autoencoder (VAE) using the MNIST dataset. 

The Quantum VAE leverages quantum computing principles to encode and decode data through quantum circuits.

## Prerequisites

Make sure you have the following packages installed:

- `pennylane`
- `torch`
- `torchvision`
- `matplotlib`

You can install them using pip:

```bash
pip install pennylane torch torchvision matplotlib
```

## Project Structure

- `VAE.py`: Contains the implementation of the QuantumVAE class.
- `test_data.py`: Example script of how to load data, train the Quantum VAE, and evaluate its performance.
- `requirements.txt`: List of required Python packages.

## QuantumVAE Class

The `QuantumVAE` class implements the following methods:

- `__init__`: Initializes the VAE with the number of qubits, latent dimensions, layers, and learning rate.
- `quantum_encoder_circuit`: Defines the quantum circuit for the encoder.
- `quantum_decoder_circuit`: Defines the quantum circuit for the decoder.
- `forward`: Performs the forward pass, encoding and decoding the input data.
- `loss`: Calculates the mean squared error loss between the input and reconstructed data.
- `train_step`: Performs a single step of optimization.
- `train`: Runs the training loop over the dataset.
- `evaluate`: Evaluates the model on the test dataset.

## Training and Evaluation

To train and evaluate the Quantum VAE, follow these steps:

1. **Load and Preprocess the MNIST Data**:

```python
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training and test sets
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
test_set = MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for the training and test sets
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

2. **Initialize and Train the VAE**:

```python
from quantum_vae import QuantumVAE

# Initialize the VAE
n_qubits = 4
vae = QuantumVAE(n_qubits=n_qubits)

# Train the VAE
vae.train(train_loader, epochs=10)
```

3. **Evaluate the VAE**:

```python
# Evaluate the VAE
vae.evaluate(test_loader)
```

4. **Plot Reconstructed Images**:

```python
import matplotlib.pyplot as plt

def plot_reconstructed_images(model, data_loader, n_images=5):
    model.encoder_qnode = qml.QNode(model.quantum_encoder_circuit, model.dev)
    model.decoder_qnode = qml.QNode(model.quantum_decoder_circuit, model.dev)
    
    data_iter = iter(data_loader)
    images, _ = data_iter.next()
    images = images.view(-1, model.n_qubits)  # Flatten the image
    
    with torch.no_grad():
        z = model.encoder_qnode(model.params_encoder, images[:n_images])
        reconstructions = model.decoder_qnode(model.params_decoder, z)
    
    fig, axes = plt.subplots(2, n_images, figsize=(10, 2))
    for i in range(n_images):
        # Original images
        ax = axes[0, i]
        ax.imshow(images[i].view(28, 28).numpy(), cmap='gray')
        ax.axis('off')
        
        # Reconstructed images
        ax = axes[1, i]
        ax.imshow(reconstructions[i].view(28, 28).numpy(), cmap='gray')
        ax.axis('off')
    
    plt.show()

# Plot reconstructed images
plot_reconstructed_images(vae, test_loader)
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```
