import pennylane as qml
from pennylane import numpy as np
import torch
from torch.optim import Adam

class QuantumVAE:
    def __init__(self, n_qubits=4, n_latent=2, n_layers=1, lr=0.01):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.lr = lr
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params_encoder = np.random.random((n_layers, 2 * n_qubits))
        self.params_decoder = np.random.random((n_layers, 2 * n_qubits))
        self.encoder_qnode = qml.QNode(self.quantum_encoder_circuit, self.dev)
        self.decoder_qnode = qml.QNode(self.quantum_decoder_circuit, self.dev)
        self.optimizer = Adam([torch.tensor(self.params_encoder, requires_grad=True), 
                        torch.tensor(self.params_decoder, requires_grad=True)], lr=lr)

    def quantum_encoder_circuit(self, params, x):
        qml.AngleEmbedding(x, wires=range(self.n_qubits))
        for i in range(self.n_layers):
            for j in range(self.n_qubits):
                qml.RX(params[i][j], wires=j)
                qml.CNOT(wires=[j, (j+1) % self.n_qubits])
                qml.RY(params[i][j + self.n_qubits], wires=j)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def quantum_decoder_circuit(self, params, z):
        qml.AngleEmbedding(z, wires=range(self.n_qubits))
        for i in range(self.n_layers):
            for j in range(self.n_qubits):
                qml.RY(params[i][j], wires=j)
                qml.CZ(wires=[j, (j+1) % self.n_qubits])
                qml.RX(params[i][j + self.n_qubits], wires=j)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        z = self.encoder_qnode(self.params_encoder, x)
        x_reconstructed = self.decoder_qnode(self.params_decoder, z)
        return x_reconstructed

    def loss(self, x, x_reconstructed):
        return torch.nn.functional.mse_loss(torch.tensor(x_reconstructed), torch.tensor(x))

    def train_step(self, x):
        self.optimizer.zero_grad()
        z = self.encoder_qnode(self.params_encoder, x)
        x_reconstructed = self.decoder_qnode(self.params_decoder, z)
        loss_value = self.loss(x, x_reconstructed)
        loss_value.backward()
        self.optimizer.step()
        return loss_value.item()

    def train(self, data, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for x, _ in data:
                x = x.view(-1, self.n_qubits)  # Flatten the image
                total_loss += self.train_step(x)
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data)}')

    def evaluate(self, data):
        total_loss = 0
        with torch.no_grad():
            for x, _ in data:
                x = x.view(-1, self.n_qubits)  # Flatten the image
                z = self.encoder_qnode(self.params_encoder, x)
                x_reconstructed = self.decoder_qnode(self.params_decoder, z)
                loss_value = self.loss(x, x_reconstructed)
                total_loss += loss_value.item()
        average_loss = total_loss / len(data)
        print(f'Evaluation Loss: {average_loss}')
        return average_loss
