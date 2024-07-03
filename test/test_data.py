import torch
import pennylane as qml
import matplotlib as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def preprocess_MNIST():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_loader = DataLoader(MNIST(root='./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST(root='./data', train=False, download=True, transform=transform), batch_size=64, shuffle=False)
    return train_loader, test_loader

def plot_reconstructed_images(model, data_loader, n_images=5):
    model.encoder_qnode = qml.QNode(model.quantum_encoder_circuit, model.dev)
    model.decoder_qnode = qml.QNode(model.quantum_decoder_circuit, model.dev)
    
    data_iter = iter(data_loader)
    images, _ = data_iter.next()
    images = images.view(-1, model.n_qubits)  # Flatten the image
    
    with torch.no_grad():
        z = model.encoder_qnode(model.params_encoder, images[:n_images])
        reconstructions = model.decoder_qnode(model.params_decoder, z)
    
    _, axes = plt.subplots(2, n_images, figsize=(10, 2))
    for i in range(n_images):
        ax = axes[0, i]
        ax.imshow(images[i].view(28, 28).numpy(), cmap='gray')
        ax.axis('off')
        ax = axes[1, i]
        ax.imshow(reconstructions[i].view(28, 28).numpy(), cmap='gray')
        ax.axis('off')
    
    plt.show()