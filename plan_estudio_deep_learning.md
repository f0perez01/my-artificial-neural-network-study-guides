# Plan de estudio para prueba de Deep Learning (4 días)

## Día 1 - Fundamentos y práctica básica con Keras
- Revisar: `01_Keras_basic.ipynb`, `02_MLP.ipynb`
- Enfocarse en:
  - Creación de modelos secuenciales
  - Uso de `compile()`, `fit()`, `evaluate()`
  - Funciones de pérdida (`loss`), optimizadores y métricas
- Tarea:
  - Escribir un resumen de 5 pasos sobre cómo se construye un modelo MLP básico

### The Sequential model
```py
import keras
from keras import layers, ops

# Define Sequetial model with 3 layers
modelo = keras.Sequential(
  [
    layers.Dense(2, activation='relu', name='layer1'),
    layers.Dense(3, activation='relu', name='layer2'),
    layers.Dense(4, name='layer3'),
  ]
)
# Call model on a test input
x = ops.ones((3,3))
y = model(x)
```

### ¿Qué caracteriza a un MLP implementado en keras?
Un MLP (Multilayer Perceptron) implementado en Keras tiene una serie de características clave que reflejan su estructura y funcionamiento como red neuronal completamente conectada:

**Características principales de un MLP en Keras**
| Característica                      | Descripción                                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Modelo secuencial**               | Se define comúnmente con `Sequential()`, ya que las capas se apilan en orden.                                |
| **Capas densas (`Dense`)**          | Cada neurona está conectada a todas las neuronas de la capa anterior. Se usa `Dense(units, activation=...)`. |
| **Entrada aplanada (`Flatten`)**    | Si trabajas con imágenes (como MNIST), se aplana con `Flatten()` antes de pasar a capas densas.              |
| **Funciones de activación**         | Usualmente `relu` para capas ocultas, `softmax` o `sigmoid` en la capa de salida (según el tipo de tarea).   |
| **Capa de salida**                  | Tiene tantas neuronas como clases (clasificación) o una sola (regresión), con activación adecuada.           |
| **Compilación del modelo**          | Requiere definir `optimizer`, `loss` y `metrics` en `model.compile(...)`.                                    |
| **Entrenamiento (`fit`)**           | Se entrena con `model.fit(x, y, epochs=n, ...)`, usando datos ya preprocesados.                              |
| **Sin convolución ni recursividad** | Es una red **completamente conectada**, no incluye capas como `Conv2D` o `LSTM`.                             |

**Ejemplo básico de MLP en Keras:**

```py

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),       # Aplana la imagen 28x28
    Dense(128, activation='relu'),       # Capa oculta
    Dense(10, activation='softmax')      # Capa de salida para clasificación en 10 clases
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

```

**Típico en tareas como:**
* Clasificación de dígitos (MNIST)
* Predicciones simples en tabulares
* Problemas donde los datos no tienen estructura espacial o secuencial relevante


---

## Día 2 - MLP y clasificación de imágenes
- Revisar: `04_MLP_FashionMNIST.ipynb`, `solución_ejercicios_propuestos_MLP.ipynb`
- Enfocarse en:
  - Preprocesamiento y normalización de datos
  - Arquitectura de un MLP
  - Overfitting y cómo enfrentarlo
- Tarea:
  - Simular respuesta a la pregunta: “¿Qué harías si el modelo sobreajusta?”

---

## Día 3 - CNN y datasets más complejos (CIFAR-10)
- Revisar: `03_CIFAR.ipynb`, `solución_ejercicios_propuestos_CNN.ipynb`
- Enfocarse en:
  - Capas `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`
  - Preprocesamiento de imágenes reales
- Tarea:
  - Escribir una CNN simple desde cero como ejercicio práctico

---

## Día 4 - Transfer Learning y simulación de prueba
- Revisar: `05_transfer_learning_and_prediction.ipynb`
- Enfocarse en:
  - Qué es transfer learning y cuándo se aplica
  - Modelos base (`MobileNet`, `VGG`, etc.)
  - Congelamiento vs entrenamiento completo
- Tarea:
  - Simulación de prueba:
    - 3 preguntas teóricas
    - Crear un modelo transfer y hacer predicción con imagen (por ejemplo: `supuestamente_un_perro.jpg`)

---

## Sugerencias adicionales
- No memorices, entiende el flujo: datos → modelo → compilación → entrenamiento → evaluación.
- Prepárate para explicar conceptos como overfitting, regularización y transfer learning.
- Enfócate en comprender código ya hecho y poder modificarlo según el problema.

