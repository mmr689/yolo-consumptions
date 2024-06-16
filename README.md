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
        <th>INT8</th>
    </tr>
    <tr>
        <td>RPi3B+</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>RPi4B+</td>
        <td>✔️</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>RPi5</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Coral USB Accelerator + RPi3B+</td>
        <td >➖</td>
        <td></td>
    </tr>
    <tr>
        <td>Coral USB Accelerator + RPi4B+</td>
        <td >➖</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>Coral USB Accelerator + RPi5</td>
        <td >➖</td>
        <td></td>
    </tr>
    <tr>
        <td>Coral Dev Board</td>
        <td >➖</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>Coral Dev Board Mini</td>
        <td>❌</td>
        <td>❌</td>
    </tr>
    <tr>
        <td>Rock4C+</td>
        <td>✔️</td>
        <td>✔️</td>
    </tr>
    <tr>
        <td>Coral USB Accelerator + RockC4+</td>
        <td >➖</td>
        <td>✔️</td>
    </tr>
</table>
