import numpy as np
import DnnLib
import matplotlib.pyplot as plt
import json

model = {}
data_train = np.load("mnist_train.npz")
data_test = np.load("mnist_test.npz")

with open("mnist_trained_mlp.json", "r") as f:
    model = json.load(f)

scale = model["preprocess"]
entries = []

train_images = data_train["images"]
train_labels = data_train["labels"]
test_images = data_test["images"]
test_labels = data_test["labels"]

entries = np.array([img.flatten() / scale["scale"] for img in train_images])
test_entries = np.array([img.flatten() / scale["scale"] for img in test_images])


def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


num_classes = model["layers"][-1]["units"]
one_hot_labels_train = one_hot_encode(train_labels, num_classes)
one_hot_labels_test = one_hot_encode(test_labels, num_classes)

layers = []
input_size = entries.shape[1]

for item in model["layers"]:
    activation = None
    if item["activation"] == "relu":
        activation = DnnLib.ActivationType.RELU
    elif item["activation"] == "softmax":
        activation = DnnLib.ActivationType.SOFTMAX

    output_size = item["units"]
    layer = DnnLib.DenseLayer(input_size, output_size, activation)
    layer.weights = np.array(item["W"]).T
    layer.bias = np.array(item["b"])
    layers.append(layer)
    input_size = output_size


def forward_pass(network, input_data):
    current_output = input_data
    for layer in network:
        current_output = layer.forward(current_output)
    return current_output

print(len(test_entries))
print(test_entries)

predictions = forward_pass(layers, test_entries)

def categorical_cross_entropy(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # The loss formula
    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
    return loss


loss = categorical_cross_entropy(one_hot_labels_test, predictions)
print(f"Categorical Cross-Entropy Loss: {loss:.4f}")

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(one_hot_labels_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_images[i], cmap="gray")
    true_label = true_classes[i]
    pred_label = predicted_classes[i]
    title = f"True: {true_label}\nPred: {pred_label}"
    color = "green" if true_label == pred_label else "red"
    plt.title(title, color=color)
    plt.axis("off")
plt.tight_layout()
plt.show()
