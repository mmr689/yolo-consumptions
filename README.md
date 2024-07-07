<p align="center">
  <img src="static/banner10.png" alt="Banner" width="100%">
</p>

<h1>YOLO: Energy consumption evaluation on Raspberry Pi and Coral platforms</h1>

<h2>Devices</h2>
<ul>
  <li>RPi 1B</li>
  <li>RPi 3B</li>
  <li>RPi 4B</li>
  <li>RPi 5</li>
  <li>Coral USB Accelerator</li>
  <li>Coral Dev Board</li>
  <li>Coral Dev Board Mini</li>
</ul>

<h2>Tasks</h2>
<ol>
  <li>RPi FP32 Ultralytics</li>
  <li>RPi FP32 TFLite</li>
  <li>RPi INT8 TFLite</li>
  <li>RPi + Coral USB Accelerator INT8 TFLite. ¿Se pued con FP32 ultra y tf?</li>
  <li>Coral INT8 TFLite</li>
</ol>
<ul>
  <li>¿COMPARO ENTRE SOLO TRABJAR UNA O VARIAS?</li>
</ul>



- https://github.com/feranick/libedgetpu/releases
  - https://github.com/feranick/libedgetpu/releases/download/v16.0TF2.15.1-1/libedgetpu1-std_16.0tf2.15.1-1.bookworm_arm64.deb

wget https://github.com/feranick/libedgetpu/releases/download/v16.0TF2.15.1-1/libedgetpu1-std_16.0tf2.15.1-1.bookworm_arm64.deb
sudo apt install ./libedgetpu1-std_16.0tf2.15.1-1.bookworm_arm64.deb

Rock
wget https://github.com/feranick/libedgetpu/releases/download/v16.0TF2.15.1-1/libedgetpu1-std_16.0tf2.15.1-1.bullseye_arm64.deb
sudo apt install ./libedgetpu1-std_16.0tf2.15.1-1.bullseye_arm64.deb
sudo find / -name libedgetpu.so.1
> /usr/lib/aarch64-linux-gnu/libedgetpu.so.1
```python
from tflite_runtime.interpreter import Interpreter, load_delegate

# Actualiza la ruta del delegado
delegate_path = '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1'

# Ejemplo de cómo cargar el modelo con el delegado
interpreter = Interpreter(
    model_path='path_to_your_model.tflite',
    experimental_delegates=[load_delegate(delegate_path, options={'device': 'usb'})]
)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu

NO SE EL PASO PERO CONVIENE REINICIAR

# Ejecutar
- `python 06B_run-tflite.py FP32 Server`
- `python 06B_run-tflite.py FP32 Server --save_img True`

# Marcas temporales (GPIOs/timer)

- Carga del modelo.
- Inferencia


Script "test"

tf fp32
  normaliza en float.

tf int8
  normaliza int8
  redefine coordenadas para int8

tf int8 quant
  carga modelo adaptado a edgetpu
  normaliza int 8
  redefine coords para int8


# Devices

<table>
    <tr>
        <th>Devices</th>
        <th>FP32</th>
        <th>FP16</th>
        <th>INT8</th>
        <th>EdgeTPU</th>
    </tr>
    <tr>
        <td>RPi3B</td>
        <td>✔️</td>
        <td>✔️</td>
        <td>✔️</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>RPi4B</td>
        <td>✔️</td>
        <td>✔️</td>
        <td>✔️</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>RPi5B</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Coral Dev Board</td>
        <td>➖</td>
        <td>➖</td>
        <td>➖</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>Coral Dev Board Mini</td>
        <td>❌</td>
        <td>❌</td>
        <td>❌</td>
        <td></td>
    </tr>
    <tr>
        <td>Rock4C+</td>
        <td>✔️</td>
        <td>✔️</td>
        <td>✔️</td>
        <td>✔️</td>
    </tr>
</table>

<h2>mAP Scores for Different Model Precisions on BioVIEW lizards dataset</h2>

<table>
    <thead>
        <tr>
            <th>Precisión</th>
            <th>mAP<sub>50</sub></th>
            <th>mAP<sub>75</sub></th>
            <th>mAP<sub>95</sub></th>
            <th>mAP<sub>50-95</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FP32</td>
            <td>97.24%</td>
            <td>86.90%</td>
            <td>2.07%</td>
            <td>72.62%</td>
        </tr>
        <tr>
            <td>FP16</td>
            <td>97.24%</td>
            <td>86.21%</td>
            <td>2.07%</td>
            <td>72.76%</td>
        </tr>
        <tr>
            <td>INT8</td>
            <td>0.13%</td>
            <td>0.00%</td>
            <td>0.00%</td>
            <td>0.02%</td>
        </tr>
    </tbody>
</table>

<h2>Comprehensive performance comparison across different device configurations and precision types</h2>
<table border="1">
    <thead>
        <tr>
            <th>Device</th>
            <th>Model Type</th>
            <th>Model Load Time (s)</th>
            <th>Average Image Prediction Time (s)</th>
            <th>Total Execution Time (s)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Rock4Plus (FP32)</td>
            <td>FP32</td>
            <td>0.413</td>
            <td>1.062</td>
            <td>22.077</td>
        </tr>
        <tr>
            <td>RPi4B (FP32)</td>
            <td>FP32</td>
            <td>0.404</td>
            <td>1.065</td>
            <td>21.053</td>
        </tr>
        <tr>
            <td>RPi3B (FP32)</td>
            <td>FP32</td>
            <td>0.937</td>
            <td>4.099</td>
            <td>85.388</td>
        </tr>
        <tr>
            <td>Rock4Plus (INT8)</td>
            <td>INT8</td>
            <td>0.069</td>
            <td>0.754</td>
            <td>18.930</td>
        </tr>
        <tr>
            <td>RPi4B (INT8)</td>
            <td>INT8</td>
            <td>0.080</td>
            <td>0.667</td>
            <td>15.894</td>
        </tr>
        <tr>
            <td>RPi3B (INT8)</td>
            <td>INT8</td>
            <td>0.151</td>
            <td>3.100</td>
            <td>60.622</td>
        </tr>
        <tr>
            <td>DevBoard (INT8)</td>
            <td>INT8</td>
            <td>0.056</td>
            <td>0.105</td>
            <td>16.034</td>
        </tr>
        <tr>
            <td>Rock-USBCoral (INT8)</td>
            <td>INT8 + Coral</td>
            <td>5.213</td>
            <td>0.115</td>
            <td>18.176</td>
        </tr>
        <tr>
            <td>RPi4B-USBCoral (INT8)</td>
            <td>INT8 + Coral</td>
            <td>2.671</td>
            <td>0.084</td>
            <td>12.583</td>
        </tr>
        <tr>
            <td>RPi3B-USBCoral (INT8)</td>
            <td>INT8 + Coral</td>
            <td>4.641</td>
            <td>0.529</td>
            <td>49.029</td>
        </tr>
        <tr>
            <td>Rock4Plus (FP16)</td>
            <td>FP16</td>
            <td>0.093</td>
            <td>1.078</td>
            <td>21.922</td>
        </tr>
        <tr>
            <td>RPi4B (FP16)</td>
            <td>FP16</td>
            <td>0.335</td>
            <td>0.925</td>
            <td>18.474</td>
        </tr>
        <tr>
            <td>RPi3B (FP16)</td>
            <td>FP16</td>
            <td>0.749</td>
            <td>3.203</td>
            <td>82.481</td>
        </tr>
    </tbody>
</table>

<h3>Conclusions</h3>
<ul>
    <li><strong>Model Load Time:</strong> The devices show varied model load times, with USB Coral equipped devices generally having longer load times due to additional initializations required for the accelerators.</li>
    <li><strong>Image Prediction Performance:</strong> INT8 and FP16 models generally provide faster image prediction times compared to FP32, demonstrating the efficiency of using quantized models for faster inference without significant loss of accuracy.</li>
    <li><strong>Total Execution Time:</strong> Devices using lower precision types (INT8, FP16) tend to have shorter total execution times, illustrating their suitability for applications requiring high efficiency and rapid processing.</li>
    <li><strong>Impact of Hardware Accelerators:</strong> The addition of USB Coral significantly improves performance, especially notable in devices like the RPi4B-USBCoral, which shows one of the shortest total execution times and the fastest average image prediction times across all configurations.</li>
    <li><strong>Hardware Capabilities:</strong> Newer and more advanced devices (e.g., Rock4Plus and DevBoard) perform consistently well across different precision types, suggesting that hardware upgrades can effectively boost performance for demanding AI tasks.</li>
</ul>


# mAP

El **mAP (mean Average Precision)** es una métrica comúnmente usada para evaluar modelos de detección de objetos. Para calcularlo, primero debes determinar la precisión promedio (AP) para cada clase y luego calcular la media de estos valores AP para todas las clases. A continuación te detallo los pasos y métodos para calcular el mAP:

### 1. Calcular la Precisión y el Recall para cada clase
Para cada clase de objeto que detecta tu modelo, debes calcular la precisión y el recall basado en verdaderos positivos (TP), falsos positivos (FP) y falsos negativos (FN).

- **Precisión** $P$ se define como $P = \frac{TP}{TP + FP}$
- **Recall** $R$ se define como $R = \frac{TP}{TP + FN}$

### 2. Curva Precision-Recall
Para cada clase, debes generar una curva Precision-Recall. Esto implica variar un umbral sobre las puntuaciones de detección asignadas a las predicciones para calcular diferentes valores de precisión y recall.

### 3. Calcular el Average Precision (AP) para cada clase
El AP se puede calcular como el área bajo la curva Precision-Recall. Una forma común de hacerlo es utilizando la aproximación de interpolación de 11 puntos, donde se calcula el promedio de los valores máximos de precisión para recall que supera los umbrales específicos (0, 0.1, 0.2, ..., 1).

Sin embargo, un método más moderno y común es integrar la curva utilizando todos los puntos (interpolación continua), comúnmente hecho sumando el área bajo la curva.

### 4. Calcular el mAP
Una vez que tengas el AP para cada clase, el mAP se calcula simplemente como el promedio de estos APs para todas las clases relevantes.

### Ejemplo de Código Python
Aquí hay un ejemplo simple de cómo podrías calcular la precisión y el recall para un conjunto de predicciones, y después calcular el mAP. Este ejemplo asume que tienes las listas `true_positives`, `false_positives`, y `false_negatives` para cada clase:

```python
import numpy as np

# Datos simulados: listas de TP, FP, FN para cada clase
tps = [np.array([tp_class1, tp_class2, tp_class3])]
fps = [np.array([fp_class1, fp_class2, fp_class3])]
fns = [np.array([fn_class1, fn_class2, fn_class3])]

# Función para calcular AP para una clase
def average_precision(rec, prec):
    return np.sum((rec[1:] - rec[:-1]) * prec[1:])

aps = []
for tp, fp, fn in zip(tps, fps, fns):
    # Ordenar por score aquí si tus arrays no están pre-ordenados
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Ordenar por recall (esto es crucial para el cálculo del AP)
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]

    # Asegurar que la precisión es decreciente
    for i in range(len(sorted_precision)-2, -1, -1):
        sorted_precision[i] = max(sorted_precision[i], sorted_precision[i+1])

    # Calcular AP
    ap = average_precision(sorted_recall, sorted_precision)
    aps.append(ap)

# Calcular mAP
mean_ap = np.mean(aps)
print(f"mAP: {mean_ap}")
```

### Consideraciones
- **Interpolación de la precisión**: Asegúrate de que los valores de precisión son decrecientes cuando calculas el AP.
- **Datos etiquetados**: Necesitas tener las etiquetas verdaderas y las puntuaciones de confianza para tus predicciones.
- **Librerías y herramientas**: Existen librerías que pueden simplificar este proceso, como scikit-learn en Python, que ofrece métodos para calcular directamente la curva Precision-Recall y el área bajo la curva.
