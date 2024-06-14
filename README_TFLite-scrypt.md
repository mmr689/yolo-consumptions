# Object Detection System using YOLO TFLite on Raspberry Pi

## Overview
This script configures a Raspberry Pi to perform object detection using the YOLO (You Only Look Once) model. It is designed to be used with either the Raspberry Pi's CPU, a Coral Edge TPU, or a USB-attached Coral Edge TPU. The script handles image processing, object detection, and logging of detection timings. GPIO pins are used to signal the start and end of significant operations, which can be monitored with an oscilloscope for debugging or performance measurement purposes.

## Features
- Object detection using YOLO TFLite models.
- Supports different precision models (FP32, INT8 Quantized).
- Compatible with Raspberry Pi models 3, 4, and 5, as well as Coral Edge TPUs, Coral Dev Board, Coral Dev Board Mini, and Coral USB Accelerator.
- Timing measurements for model loading and image processing.
- Error logging and operational signals via GPIO.

## Prerequisites
- Python 3.11.2
- TensorFlow Lite Runtime
- OpenCV, NumPy, and other required Python libraries
- Coral Edge TPU (optional, for accelerated inference)
- Properly formatted TFLite models placed in a known directory

## Installation
1. **Prepare your Raspberry Pi**: Ensure that your Raspberry Pi is set up with the latest version of Raspberry Pi OS and that it has internet access.

2. **Install Python and pip**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

3. **Install Required Libraries**:
   ```bash
   pip3 install numpy opencv-python-headless
   ```

   For the TensorFlow Lite runtime, follow the [official installation guide](https://www.tensorflow.org/lite/guide/python).

4. **Hardware Setup**:
   Connect your Coral Edge TPU via USB if applicable. Ensure that GPIO pins are accessible if using GPIO signaling.

## Usage
To run the script, you need to specify the model precision and the device as command-line arguments.

```bash
python3 path_to_script.py [precision] [device]
```

Where:
- `[precision]` is either `FP32` or `INT8`.
- `[device]` is `RPi`, `EdgeTPU`, or `USB-EdgeTPU`.

Example:
```bash
python3 06B_run-tflite.py FP32 RPi
```

## Configuration
Modify the `main` function in the script to point to your specific model files and image directories as needed.

## Output
The results of the object detection will be saved in the specified results directory, and timings will be logged in a JSON file as well as outputted to a specified log file.

## Contributing
Contributions to this project are welcome. Please ensure that you test the changes on your hardware setup and follow the existing coding style.

## License
Specify the license under which your project is made available. (e.g., MIT, GPL-3.0, etc.)

## Authors
- Your Name or Your Organization