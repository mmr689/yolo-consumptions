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