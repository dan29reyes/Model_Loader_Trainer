# 🧠 Prueba de Red Neuronal para MNIST

Este proyecto presenta pruebas de entrenamiento de un modelo de red neuronal utilizando el conjunto de datos **MNIST** y **Fashion MNIST**.  
Se han experimentado diferentes configuraciones de épocas, tamaño de batch y tasa de aprendizaje.

---

## 📊 Pruebas y Resultados

### 🔹 Numbers MNIST – 10 épocas
- Épocas: `10`
- Batch Size: `128`
- Learning Rate: `0.001`

**Gráfico de Entrenamiento:**

![t-epoch-10-nm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/train-10-epochs-mnist.png?raw=true)

**Resultados del Modelo:**

![r-epoch-10-nm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/results-10-epochs-mnist.png?raw=true)

---

### 🔹 Numbers MNIST – 5 épocas
- Épocas: `5`
- Batch Size: `128`
- Learning Rate: `0.001`

**Gráfico de Entrenamiento:**

![t-epoch-5-nm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/train-5-epochs-mnist.png?raw=true)

**Resultados del Modelo:**

![r-epoch-5-nm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/results-5-epochs-mnist.png?raw=true)

---

### 🔹 Fashion MNIST – 10 épocas
- Épocas: `10`
- Batch Size: `128`
- Learning Rate: `0.001`

**Gráfico de Entrenamiento:**

![t-epoch-10-fm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/train-10-epochs-fashion-mnist.png?raw=true)

**Resultados del Modelo:**

![r-epoch-10-fm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/results-10-epochs-fashion-mnist.png?raw=true)

---

### 🔹 Fashion MNIST – 5 épocas
- Épocas: `5`
- Batch Size: `128`
- Learning Rate: `0.001`

**Gráfico de Entrenamiento:**

![t-epoch-5-fm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/train-5-epochs-fashion-mnist.png?raw=true)

**Resultados del Modelo:**

![r-epoch-5-fm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/results-5-epochs-fashion-mnist.png?raw=true)

---

## ⭐ Pruebas Extra ⭐
Pruebas Realizadas para medir la precisión del modelo cuando se utilizan datasets diferentes, se probo entrenar con Fashion MNIST y evaluar con Numbers MNIST y viceversa.

### 🔹 Numbers MNIST – 10 épocas
- Épocas: `10`
- Batch Size: `128`
- Learning Rate: `0.001`
- Entrenamiento: `Fashion MNIST`
- Evaluación: `Numbers MNIST`

**Resultados del Modelo:**

![r-epoch-10-nm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/test-fashion-train-result-numbers.png?raw=true)

---

### 🔹 Fashion MNIST – 10 épocas
- Épocas: `10`
- Batch Size: `128`
- Learning Rate: `0.001`
- Entrenamiento: `Numbers MNIST`
- Evaluación: `Fashion MNIST`

**Resultados del Modelo:**

![r-epoch-10-fm](https://github.com/dan29reyes/Model_Loader_Trainer/blob/main/train_images/test-numbers-train-result-fashion.png?raw=true)

---

## 👨‍💻 Autor
Programa diseñado por **Kenneth Daniel Reyes**  