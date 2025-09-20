import DnnLib
import numpy as np

def create_network(model_config, input_size):
    layers = []
    dropout_layers = []
    
    for item in model_config['layers']:
        activation_type = None
        if item['activation'] == "relu":
            activation_type = DnnLib.ActivationType.RELU
        elif item['activation'] == "softmax":
            activation_type = DnnLib.ActivationType.SOFTMAX
        
        output_size = item['units']
        layer = DnnLib.DenseLayer(input_size, output_size, activation_type)
        layers.append(layer)
        input_size = output_size
        
        if item['activation'] != "softmax":
            dropout = DnnLib.Dropout(dropout_rate=0.5)
            layers.append(dropout)
            dropout_layers.append(dropout)
            
    return layers, dropout_layers

def fwd_pass(layers, x, training=True):
    activation = x
    for layer in layers:
        if hasattr(layer, 'training'):
            layer.training = training
            activation = layer.forward(activation)
        else:
            activation = layer.forward(activation)
    return activation

def bwd_pass(layers, grad_output):
    grad = grad_output
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return grad