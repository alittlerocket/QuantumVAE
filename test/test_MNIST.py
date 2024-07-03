from test_data import *
from ..quantum_vae.VAE import QuantumVAE

if __name__ == "__main__":
    train_loader, test_loader = preprocess_MNIST()
    vae=QuantumVAE(n_qubits=4)
    vae.train(train_loader, epochs=10)
    vae.evaluate(test_loader)
    plot_reconstructed_images(vae, test_loader)
    




