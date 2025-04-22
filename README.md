# Seguimiento y Análisis de Movimiento Armónico de un Péndulo con Visión por Computador

Este proyecto utiliza técnicas de procesamiento de imágenes y análisis de datos físicos para rastrear el movimiento de un péndulo en un video y extraer información clave como posición, velocidad, aceleración y ángulo. Está diseñado para analizar fenómenos de movimiento armónico simple (MAS) a partir de imágenes, con fines educativos o científicos.

---

## 🎥 Video de Entrada

El sistema toma como entrada un video (`long.mp4` o `short.mp4`) donde se observa un péndulo en movimiento. El código segmenta el péndulo con base en su color y rastrea su posición cuadro por cuadro.


Puede descargar el video corto [aquí](https://udeaeduco-my.sharepoint.com/:v:/r/personal/cristiany_jimenez_udea_edu_co/Documents/DPI/resources/short.mp4?csf=1&web=1&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=zEnBRy)
o descargar el video largo [aquí](https://udeaeduco-my.sharepoint.com/:v:/g/personal/cristiany_jimenez_udea_edu_co/EQsgSDERYTRKjnSTclSJR3gBvBABG9VxVrtIeNxnA_JmzQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=OmUsKt)
---

## 🧰 Tecnologías y Librerías Usadas

- `Python 3`
- `OpenCV`: Segmentación y seguimiento del péndulo
- `NumPy`: Operaciones matemáticas y de arrays
- `Matplotlib`: Visualización de resultados (usado fuera del loop principal)
- `SciPy`: Suavizado de señales y detección de picos (para análisis posterior)
- `math`: Cálculo de ángulos y magnitudes físicas

---

## 📐 Calibración y Datos Reales

- Diámetro real de la masa del péndulo: **2.5 cm**
- Longitud de la cuerda: **10.3 cm**
- El sistema realiza una **calibración automática** durante el primer frame para convertir píxeles a centímetros.

---

## ⚙️ Cómo Funciona

1. **Lectura del video** y obtención de propiedades (FPS, resolución, total de cuadros).
2. **Redimensionamiento del video** para mejorar el procesamiento.
3. **Segmentación en HSV** para detectar el péndulo (ajustar color en `lower_limit` y `upper_limit`).
4. **Detección del contorno más grande**, asumiendo que es la masa del péndulo.
5. **Cálculo del centroide y su ángulo** respecto a un punto fijo de sujeción.
6. **Conversión de coordenadas a centímetros** mediante calibración inicial.
7. **Cálculo de velocidad y aceleración** en tiempo real por diferencias finitas.
8. **Visualización en tiempo real** de los datos físicos sobre el video.
9. **Almacenamiento de datos** en estructuras para análisis posterior.

---

## 📊 Datos Almacenados

Los datos guardados por frame incluyen:

- Tiempo (`time`)
- Posición en píxeles y centímetros (`x_px`, `y_px`, `x_cm`, `y_cm`)
- Velocidades (`vx`, `vy`, `v_total`)
- Aceleraciones (`ax`, `ay`, `a_total`)
- Ángulo respecto a la vertical (`angle`)

---

## 📌 Punto Fijo

El punto de sujeción del péndulo es fijo y está definido en:

```python
anchor_x = 344
anchor_y = 0

Esto se puede modificar si cambia la posición del péndulo en el video.

## 🖼️ Ejemplo de Visualización
El sistema muestra en pantalla en tiempo real:

- Tiempo transcurrido
- Posición (X, Y)
- Velocidad total
- Aceleración total
- Ángulo respecto a la vertical
- Trayectoria del péndulo sobre la imagen

## ⚠️ Notas
- Asegúrate de ajustar los valores de segmentación (`lower_limit`, `upper_limit`) al color de tu péndulo.
- El video debe tener fondo relativamente uniforme para una mejor segmentación.
- Los datos recolectados se pueden exportar a CSV o utilizar con `Matplotlib` para graficar el movimiento.

## 🧪 Posibles Extensiones
- Exportar los datos a un archivo `.csv`.
- Suavizar la señal del ángulo con `savgol_filter` para detectar periodos y amortiguamiento.
- Calcular frecuencia angular, periodo y constante de restitución.
- Detección automática del punto de sujeción mediante `HoughLines` u otras técnicas.

## 🧑‍🔬 Autor y Créditos
Este proyecto fue desarrollado como parte de un estudio sobre el movimiento armónico simple (MAS) aplicado a péndulos usando visión por computador.

## 🗃️ Estructura del Proyecto
```
📁 proyecto_pendulo/
├── long.mp4
├── pendulo_tracking.py
└── README.md
```

## ▶️ Requisitos para Ejecutar
Instala las dependencias necesarias:
```bash
pip install opencv-python numpy matplotlib scipy
```

Luego ejecuta el script:
```bash
python pendulo_tracking.py
```

## 🧠 Conceptos Científicos Aplicados
- Movimiento armónico simple (MAS)
- Cinemática (posición, velocidad, aceleración)
- Conversión de unidades (px → cm)
- Cálculo de ángulos y vectores
