
# Aspectos Prácticos de las CNN - Deep Learning

Pontificia Universidad Católica de Chile  
Escuela de Ingeniería - Departamento de Ciencia de la Computación  
**Curso:** Deep Learning  
**Docente:** Ariel Reyes  

---

## 🧠 Tema central

**Entrenar y utilizar CNNs de forma práctica no es trivial**, por lo que se estudian tres grandes preguntas:

1. ¿Cómo entrenar y optimizar?
2. ¿Cómo reutilizar lo aprendido por una CNN?
3. ¿Cómo mejorar las CNNs para llevarlas a entornos reales?

---

## 1. 🛠️ ¿Cómo entrenar y optimizar una CNN?

### 🔍 Qué y para qué estamos optimizando

- Minimizar función de pérdida (ej. Cross Entropy Loss)
- Calcular derivadas (gradientes) y descender en la función objetivo
- Regularización explícita:

```math
\text{argmin}_W J(X,Y;W) = \lambda \mathcal{R}(W) + \sum_{i=1}^N \mathcal{L}(x_i, y_i; W)
```

### ⚠️ Desafíos del entrenamiento

- Costo computacional del gradiente completo
- Evitar mínimos locales y puntos críticos
- Ajustar learning rate global y por parámetro

### 🚀 Algoritmos de optimización

- **SGD**: Mini-batches, estocástico, eficiente
- **Momentum**: Evita estancamiento en mínimos
- **Adam**: Learning rate adaptativo + Momentum

---

## 2. 🔄 ¿Cómo reutilizar lo aprendido por una CNN?

### 📦 Transfer Learning

- Evidencia de generalización
- Features jerárquicas y composicionales

### 🧪 Ejemplo: AlexNet

- Usar como extractor de características
- Funciona bien para tareas como reconocimiento de escenas, acciones, rostros, etc.

### ⚙️ Mecanismos

- **Transferencia directa**
- **Fine-tuning**

### 🧠 Aplicaciones reales

- Estimar población desplazada (ResNet-50)
- Análisis de tráfico vehicular

### ❓ Desafíos

- ¿Qué capas reentrenar?
- ¿Fine-tuning o desde cero?
- ¿Dominios similares?
- ¿Learning rate adecuado?

---

## 3. 🌍 ¿Cómo llevar las CNNs al mundo real?

### 📊 Escalando el problema

#### ➤ Segmentación semántica

- Encoder-decoder
- Skip connections para recuperar detalles

#### ➤ Detección de objetos

- **Faster R-CNN** (dos etapas)
- **YOLO, SSD, RetinaNet** (una etapa)

#### ➤ Segmentación por instancia

- **Mask R-CNN**: Añade predicción de máscara por objeto

#### ➤ Dense Captioning

- Red que detecta objetos y genera descripciones simultáneamente

---

## 📚 Ejercicios propuestos

### Parte 1: Transfer Learning

- Entrenar CNN en CIFAR10 o CIFAR100
- Evaluar en dataset de gatos y perros
- Comparar con red preentrenada en ImageNet

### Parte 2: Autoencoder

- Entrenar Autoencoder convolucional
- Evaluar transferencia del encoder
- Comparar con los enfoques anteriores

---

## 📝 Comentarios finales

- Entrenar redes CNN implica múltiples decisiones prácticas
- Podemos aprovechar trabajos previos
- Las CNN pueden adaptarse a diversos problemas de visión por computador
- Las arquitecturas combinadas y multitarea permiten avanzar en comprensión de imágenes

---
