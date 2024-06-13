Referencia pone que es entre python 3.6 y 3.9. Yo tengo 3.11.2, así que ya veremos.... <<<<<<< NO CREO, esa condicición és para PyCoral y yo trabajo con TFLite

## 1: Install the Edge TPU runtime
1. Add our Debian package repository to your system
```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
```
2. Install the Edge TPU runtime:
```bash
sudo apt-get install libedgetpu1-std
```

3. Now connect the USB Accelerator to your computer using the provided USB 3.0 cable. If you already plugged it in, remove it and replug it so the newly-installed `udev` rule can take effect.

## 2: Install the PyCoral library

```bash
sudo apt-get install python3-pycoral
```