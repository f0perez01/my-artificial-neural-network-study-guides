# 📚 Apuntes complementados: Redes Neuronales Recurrentes (RNN)

## 1️⃣ Motivación: por qué RNN
**Problema**: Queremos procesar datos que vienen en **secuencia** (texto, series temporales, audio, video, sensores).  
Ejemplos:
- Clasificar el **modo de transporte** de una persona a partir de datos de GPS.
- Analizar el **sentimiento** en una reseña.
- Generar **descripciones automáticas** para una imagen o un video.

**Limitación de redes tradicionales (MLP, CNN)**:
- Procesan cada dato de forma **independiente**, como si no estuviera relacionado con los demás.
- No tienen memoria del contexto anterior.

---

## 2️⃣ Qué veremos para resolverlo
- **Recurrencia** para modelar el estado de una secuencia:  
  - **RNN** (Red Neuronal Recurrente básica).  
  - **LSTM** (*Long Short-Term Memory*).  
  - **GRU** (*Gated Recurrent Unit*).
- **Modelos de lenguaje**:
  - Representaciones de palabras (**word2vec**).
  - Traducción automática y procesamiento de secuencias (**seq2seq**).
  - Mejoras con mecanismos de **atención**.
- **Modelos fundacionales**:
  - **Transformer**, **BERT**, **GPT**, etc.

---

## 3️⃣ Concepto central de RNN
**Idea**: procesar los elementos de la secuencia **uno por uno**, manteniendo un **estado oculto** que actúa como **memoria**.

**Analogía**: como leer un párrafo palabra por palabra, recordando lo que ya leíste.

**Fórmula típica**:
```
h_t = g(W_{xh}x_t + W_{hh}h_{t-1} + b)
```
Donde:
- `x_t`: entrada en el paso `t`.
- `h_t`: estado oculto en el paso `t` (resumen de todo lo visto hasta ese momento).
- `g`: función de activación (tanh, sigmoide).
- `W_{xh}`: pesos que transforman la entrada.
- `W_{hh}`: pesos que actualizan la memoria a partir del estado anterior.

**Ventaja**: los **parámetros se comparten** para todos los pasos, así el modelo no crece en tamaño aunque la secuencia sea muy larga.

---

## 4️⃣ Aplicaciones comunes
- Seguimiento de objetos en videos.
- Predicción de sentimiento en texto (**RNN bidireccional**).
- Generación automática de texto.
- **Image captioning**: combinación CNN (procesa imagen) + RNN (genera la frase).

---

## 5️⃣ Entrenamiento
Se usa **Backpropagation Through Time (BPTT)**:
1. “Desenrollar” la RNN en el tiempo (como si fueran muchas capas).
2. Calcular la pérdida en toda la secuencia.
3. Propagar el error hacia atrás paso a paso.

**Problemas**:
1. **Costo computacional alto** para secuencias largas → solución: **truncated backpropagation**.
2. **Vanishing gradient**: la red “olvida” dependencias lejanas.
3. **Exploding gradient**: el gradiente crece demasiado, causando inestabilidad.

---

## 6️⃣ Soluciones a problemas de gradiente
- **GRU**:
  - **Compuerta de actualización**: decide cuánto olvidar y cuánto agregar.
  - **Compuerta de reseteo**: decide qué parte de la memoria usar.
- **LSTM**:
  - Mantiene una **celda de memoria** separada del estado oculto.
  - Mejor control del flujo del gradiente.

---

## 7️⃣ Conclusiones clave
- Las RNN **modelan secuencias** de cualquier longitud gracias a su memoria interna.
- Son muy versátiles y se combinan bien con otras redes.
- Para secuencias largas, **LSTM y GRU** son preferidas porque evitan vanishing/exploding gradient.
- Hoy en día, muchos problemas secuenciales se resuelven también con **Transformers**.

---

📌 **Lecturas recomendadas**:  
- [Understanding LSTM Networks – Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
- [Exploring LSTMs – Edwin Chen](http://blog.echen.me/2017/05/30/exploring-lstms/)  
- [The Unreasonable Effectiveness of RNNs – Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)  
