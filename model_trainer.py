import DnnLib
import numpy as np
from network import create_network
from utils import one_hot_encode, train_net
import json
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(description="Train a multi-layer perceptron on the MNIST dataset.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("--batch_size", type=int, default=64, help="Size of the training batches.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
args = parser.parse_args()

start_time = time.time()

num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

train_data = np.load("mnist_train.npz")
test_data = np.load("mnist_test.npz")

with open("mnist_untrained_mlp.json", "r") as f:
    model_config = json.load(f)

scale = model_config['preprocess']

train_images = train_data['images']
train_labels = train_data['labels']
test_images = test_data['images']
test_labels = test_data['labels']

test_loss_history = []
accuracy_history = []

train_entries = np.array([img.flatten() / scale['scale'] for img in train_images])
test_entries = np.array([img.flatten() / scale['scale'] for img in test_images])

num_classes = model_config['layers'][-1]['units']
one_hot_train_labels = one_hot_encode(train_labels, num_classes)
one_hot_test_labels = one_hot_encode(test_labels, num_classes)

layers, dropout_layers = create_network(model_config, train_entries.shape[1])
for layer in layers:
    if isinstance(layer, DnnLib.DenseLayer):
        layer.set_regularizer(DnnLib.RegularizerType.L2, 0.001)

optimizer = DnnLib.Adam(learning_rate=learning_rate)

print("Iniciando entrenamiento...")
train_net(
    layers=layers,
    dropout_layers=dropout_layers,
    optimizer=optimizer,
    X_train=train_entries,
    y_train=one_hot_train_labels,
    X_test=test_entries,
    y_test=one_hot_test_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    accuracy_save=accuracy_history,
    test_loss_arr=test_loss_history
)

end_time = time.time()
training_duration = end_time - start_time
print(f"Tiempo de entrenamiento: {training_duration:.2f} segundos!")

fig, ax = plt.subplots()
ax.plot(accuracy_history)
ax.set_xlabel("Epocas")
ax.set_ylabel("Precisión (%)")
ax.set_title("Precisión")
ax.set_ylim(0, 100)
ax.grid(True)
plt.show()

trained_model = {}
trained_model['preprocess'] = model_config['preprocess']
trained_model['layers'] = []
dense_layer_idx = 0

for i in range(len(layers)):
    if isinstance(layers[i], DnnLib.DenseLayer):
        layer_info = {
            'layer': i,
            'units': layers[i].weights.shape[0],
            'activation': model_config['layers'][dense_layer_idx]['activation'],
            'W': layers[i].weights.tolist(),
            'b': layers[i].bias.tolist()
        }
        trained_model['layers'].append(layer_info)
        dense_layer_idx += 1

with open('mnist_trained_mlp.json', 'w') as f:
    json.dump(trained_model, f, indent=4)

print("\nLos pesos y sesgos entrenados se han guardado en 'mnist_trained_mlp.json'.")