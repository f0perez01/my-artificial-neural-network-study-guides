# Plan de estudio para prueba de Deep Learning (4 días)

## Día 1 - Fundamentos y práctica básica con Keras
- Revisar: `01_Keras_basic.ipynb`, `02_MLP.ipynb`
- Enfocarse en:
  - Creación de modelos secuenciales
  - Uso de `compile()`, `fit()`, `evaluate()`
  - Funciones de pérdida (`loss`), optimizadores y métricas
- Tarea:
  - Escribir un resumen de 5 pasos sobre cómo se construye un modelo MLP básico

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

