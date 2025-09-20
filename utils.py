import DnnLib
import numpy as np

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def train_net(layers, dropout_layers, optimizer, X_train, y_train, X_test, y_test, epochs, batch_size, accuracy_save, test_loss_arr):
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            output = fwd_pass_with_dropout(layers, X_batch, training=True)
            
            data_loss = DnnLib.cross_entropy(output, y_batch)
            total_reg_loss = 0.0
            for layer in layers:
                if isinstance(layer, DnnLib.DenseLayer):
                    total_reg_loss += layer.compute_regularization_loss()
            
            total_loss = data_loss + total_reg_loss
            
            grad = DnnLib.cross_entropy_gradient(output, y_batch)
            bwd_pass_with_dropout(layers, grad)
            
            for layer in layers:
                if not hasattr(layer, 'training'):
                    optimizer.update(layer)
            
            epoch_loss += total_loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        val_output = fwd_pass_with_dropout(layers, X_test, training=False)
        val_loss = DnnLib.cross_entropy(val_output, y_test)
        
        predicted_classes = np.argmax(val_output, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)

        accuracy_save.append(accuracy * 100)
        test_loss_arr.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_loss:.4f} | Validation Loss: {val_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

def fwd_pass_with_dropout(layers, x, training=True):
    activation = x
    for layer in layers:
        if hasattr(layer, 'training'):
            layer.training = training
            activation = layer.forward(activation)
        else:
            activation = layer.forward(activation)
    return activation

def bwd_pass_with_dropout(layers, grad_output):
    grad = grad_output
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return grad