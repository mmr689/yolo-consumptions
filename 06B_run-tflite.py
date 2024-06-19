"""
Object Detection System using YOLO TFLite models (FP32 & INT8)

This script configures different Raspberry Pi, Rock and Coral devices to perform object detection on a set of images using the YOLO (You Only Look Once) TFLite FP32 model.
It includes functions for setting up logging, monitoring resources, loading the YOLO TFLite model, processing images to detect objects,
and saving timing data for analysis. GPIO pins are used to signal the start and end of significant operations,
which can be monitored with an oscilloscope for debugging or performance measurement.
"""

import os  # OS operations
import time  # Time handling and measurement
import json  # JSON file reading and writing
import csv  # CSV file reading and writing
import logging  # Logging events and messages
from datetime import datetime  # Date and time handling
import argparse  # Command-line argument parsing
from multiprocessing import Value  # Sharing data between processes
import threading  # Thread handling for concurrent tasks

import psutil  # System resource monitoring
import cv2  # Image processing
import numpy as np  # Mathematical operations and multi-dimensional arrays
from tflite_runtime.interpreter import Interpreter  # Interpreter for TFLite models

def get_rpi_version():
    """
    Retrieve the version of the Raspberry Pi.

    This function reads the model information from the file '/proc/device-tree/model',
    which contains a string describing the model of the Raspberry Pi. If the model is
    known, it returns a simplified string. Otherwise, it returns
    the model information with spaces replaced by underscores. If the file is not found,
    it returns a message indicating that the model cannot be determined.

    Returns:
        str: The Raspberry Pi version or an error message.
    """
    try:
        with open("/proc/device-tree/model", "r") as file:
            model_info = file.read().strip()

        if 'Raspberry Pi 3 Model B' in model_info:
            version = 'RPi3B'
        elif 'Raspberry Pi 4 Model B' in model_info:
            version = 'RPi4B'
        elif 'Raspberry Pi 5 Model B' in model_info:
            version = 'RPi5B'
        
        else:
            version = model_info.replace(' ', '_')
    except FileNotFoundError:
        version = 'unknown_rpi'
        
    return version

def working_paths(precision, device):
    """
    Determine and create the working paths and model paths based on the model precision and device.

    Args:
        precision (str): Model precision. Can be 'FP32' or 'INT8'.
        device (str): Device to be used. Options are 'RPi', 'Rock', 'EdgeTPU', 'RPi-EdgeTPU', and 'Rock-EdgeTPU'.

    Returns:
        tuple: A tuple containing the results path and model path.
    """
    work_path = f'yolov8_{precision}_TFLite'
    if precision == 'FP32' or 'FP16':
        if device == 'RPi':
            work_path += f'_{get_rpi_version()}'
        elif device == 'Rock':
            work_path += '_Rock4Plus'
        if precision == 'FP32':
            model_path = 'best_float32.tflite'
        else:
            model_path = 'best_float16.tflite'
    elif precision == 'INT8' and (device == 'RPi' or device == 'Rock'):
        if device == 'RPi':
            work_path += f'_{get_rpi_version()}'
        elif device == 'Rock':
            work_path += '_Rock4Plus'
        model_path = 'best_full_integer_quant.tflite'
    elif precision == 'INT8' and 'EdgeTPU' in device:
        if device == 'RPi-EdgeTPU':
            work_path += f'_{get_rpi_version()}-USBCoral'
        elif device == 'Rock-EdgeTPU':
            work_path += '_Rock-USBCoral'
        else:
            work_path += '_DevBoard'
        model_path = 'best_full_integer_quant_edgetpu.tflite'
    results_path = f'results/{work_path}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return results_path, model_path

def set_gpio(device):
    """
    Configure and return the GPIO interface based on the device.

    Args:
        device (str): Device to be used. Options are 'RPi', 'EdgeTPU', 'RPi-EdgeTPU', 'Rock', and 'Rock-EdgeTPU'.

    Returns:
        GPIO object or None: Configured GPIO interface or None if using RPi GPIO.
    """
    if device == 'RPi' or device == 'RPi-EdgeTPU':
        import RPi.GPIO as gpio
        gpio.setmode(gpio.BCM) # Setup GPIO mode
        gpio_flag = True
        
    elif device == 'EdgeTPU' or device == 'Rock' or device == 'Rock-EdgeTPU':
        from periphery import GPIO
        if device == 'EdgeTPU':
            gpio = GPIO("/dev/gpiochip2", 13, "out")
        elif device == 'Rock' or device == 'Rock-EdgeTPU':
            gpio = GPIO("/dev/gpiochip3", 15, "out")
        gpio_flag = False
    
    return gpio, gpio_flag

def monitor_resources(csv_file_path, interval=1, stop_event=threading.Event(), state_marker=0):
    """
    Monitors system resources and writes to a CSV file, including a state marker.
    """
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'cpu_usage_percent', 'memory_usage_percent', 'state'])
        
        while not stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, cpu_usage, memory_usage, state_marker.value])
            time.sleep(interval)

def calculate_overlap(rect1, rect2):
    """
    Calculate the overlap between two rectangles.
    
    Args:
        rect1 (tuple): (x1, y1, x2, y2, confidence) for rectangle 1.
        rect2 (tuple): (x1, y1, x2, y2, confidence) for rectangle 2.

    Returns:
        float: The fraction of the overlap area relative to the smallest rectangle area.
    """
    # Calculate areas of the rectangles
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    # Calculate intersection
    intersection_x = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    intersection_y = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    area_intersection = intersection_x * intersection_y

    # Calculate overlap as a fraction of the smaller area
    overlap = area_intersection / min(area_rect1, area_rect2)
    return overlap

def remove_overlaps(rectangles):
    """
    Remove overlapping rectangles based on a high overlap threshold.

    Args:
        rectangles (list): List of rectangle tuples (x1, y1, x2, y2, confidence).

    Returns:
        list: List of rectangles that were removed due to overlaps.
    """
    eliminated_rectangles = []
    i = 0
    while i < len(rectangles):
        j = i + 1
        while j < len(rectangles):
            if calculate_overlap(rectangles[i], rectangles[j]) > 0.9:
                eliminated_rectangles.append(rectangles[j])
                del rectangles[j]
            else:
                j += 1
        i += 1
    return eliminated_rectangles

def setup_logging(log_path, log_to_console=True):
    """
    Set up logging configuration.

    Args:
        log_path (str): Path to save the log file.
        log_to_console (bool): If True, logs will also be printed to the console.
    """
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    
    handlers = [logging.FileHandler(log_path, mode='w')]
    if log_to_console:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_model(model_path, device, gpio, gpio_flag):
    """
    Load a YOLO model from a specified path and measure load time.
    
    Args:
        model_path (str): File path of the YOLO model.
        device (str): Device we are going to use. Can be RPi, Rock, EdgeTPU, RPi-EdgeTPU, Rock-EdgeTPU
        gpio: GPIO interface for controlling hardware signals. Can be None or a GPIO object.
        gpio_flag: (bool) Flag between RPi.GPIO and python-periphery libraries.

    Returns:
        tuple: (model, load_time) where model is the loaded YOLO model, and load_time is the time taken to load the model in seconds.
    """
    if gpio_flag:
        gpio.output(17, gpio.HIGH) # Signal GPIO pin before loading model.
    else:
        gpio.write(True)
    start_time = time.time()
    if device == 'RPi' or device == 'Rock':
        model = Interpreter(model_path=model_path)
    elif device == 'EdgeTPU':
        from tflite_runtime.interpreter import load_delegate
        model = Interpreter(model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'pci:0'})])
    elif device == 'RPi-EdgeTPU':
        from tflite_runtime.interpreter import load_delegate
        model = Interpreter(model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'usb'})])
    elif device == 'Rock-EdgeTPU':
        from tflite_runtime.interpreter import load_delegate
        model = Interpreter(model_path,
                experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1', options={'device': 'usb'})])
    model.allocate_tensors()
    end_time = time.time()
    if gpio_flag:
        gpio.output(17, gpio.LOW) # Signal GPIO pin after loading model.
    else:
        gpio.write(False)
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time

def process_images(model, imgs_path, results_path, precision, gpio, gpio_flag, bb_conf=0.5):
    """
    Process images for object detection and measure processing time.

    Args:
        model: Loaded TFLite model interpreter.
        imgs_path (str): Directory containing images to process.
        results_path (str): Directory containing images results.
        precision (str): Model precision. Can be FP32 or INT8.
        gpio: GPIO interface for controlling hardware signals. Can be None or a GPIO object.
        gpio_flag: (bool) Flag between RPi.GPIO and python-periphery libraries.
        bb_conf (float): Confidence threshold for bounding box predictions.

    Returns:
        list: A list of dictionaries, each containing the filename and the time taken to process that file.
    """

    # Obtain model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    _, model_height, model_width, _ = input_details[0]['shape']
    
    # Work with images
    image_timings = []
    for filename in os.listdir(imgs_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Obtain image
            img_path = os.path.join(imgs_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image {filename}")
                continue
            
            # Adapt image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (model_width, model_height))
            if precision == 'FP32' or 'FP16':
                img_norm = img_resized.astype(np.float32) / 255.0
            # elif precision == 'FP16':
            #     print('heey')
            #     img_norm = img_resized.astype(np.float16) / 255.0
            elif precision == 'INT8':
                img_norm = img_resized.astype(np.int8)
            img_batch = np.expand_dims(img_norm, axis=0)
            
            # Predict
            if gpio_flag:
                gpio.output(17, gpio.HIGH) # Signal GPIO pin before prediction.
            else:
                gpio.write(True)
            start_time = time.time()
            model.set_tensor(input_details[0]['index'], img_batch)
            model.invoke()
            results = model.get_tensor(output_details[0]['index'])
            end_time = time.time()
            if gpio_flag:
                gpio.output(17, gpio.LOW) # Signal GPIO pin after prediction.
            else:
                gpio.write(False)
            image_timings.append({filename: end_time - start_time})
            # Obtain all bounding boxes and confidences
            bb_dict = {}
            for i in range(output_details[0]['shape'][2]):
                confs = results[0][4:, i].flatten()
                if precision == 'FP32' or 'FP16':
                    conf, label = np.max(confs), np.argmax(confs)
                elif precision == 'INT8':
                    conf, label = (np.max(confs)+128)/255, np.argmax(confs)
                if conf > bb_conf:
                    x, y, w, h = results[0][:4, i].flatten()  # COORDS
                    if precision == 'INT8':
                        x, y, w, h = (x+128)/255, (y+128)/255, (w+128)/255, (h+128)/255

                    x, y = int(x * img.shape[1]), int(y * img.shape[0]) 
                    width, height = int(w * img.shape[1]), int(h * img.shape[0])

                    x1, y1 = x - width // 2, y - height // 2
                    x2, y2 = x1 + width, y1 + height
                    
                    if label not in bb_dict:
                        bb_dict[label] = [(x1, y1, x2, y2, conf)]
                    else:
                        bb_dict[label].append((x1, y1, x2, y2, conf))
            # Apply NMS and draw bounding boxes
            for _, vals in bb_dict.items():
                vals = sorted(vals, key=lambda x: x[4], reverse=True)
                while True:
                    previous_count = len(vals)
                    _ = remove_overlaps(vals)
                    current_count = len(vals)
                    if previous_count == current_count:
                        break

                for rectangle in vals:
                    x1, y1, x2, y2, conf = rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, str(round(conf, 1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Save the image
            save_path = os.path.join(results_path, filename)
            cv2.imwrite(save_path, img)
            logging.info(f"Processed {filename} and saved to {save_path}")

    return image_timings

def main(precision, device, results_path, model, gpio, gpio_flag, state_marker):
    """
    Main function to initialize logging, load the model, and process images.
    
    Args:
        precision (str): Model precision. Can be 'FP32' or 'INT8'.
        device (str): Device to be used for running the model. Options are 'RPi', 'Rock', 'EdgeTPU', 'RPi-EdgeTPU', and 'Rock-EdgeTPU'.
        results_path (str): Path to save the results and timings.
        model (str): Path to the model to be loaded.
        gpio: GPIO interface for controlling hardware signals. Can be None or a GPIO object.
        gpio_flag: (bool) Flag between RPi.GPIO and python-periphery libraries.
        state_marker (multiprocessing.Value): Shared integer used for tracking the state of the process.
            - 0: Initial state.
            - 1: Model loaded.
            - 2: Detection process started.
            - 3: Image processing started.
    """
    # Starting oscilloscope flag
    state_marker.value = 0
    if gpio_flag:
        gpio.setup(17, gpio.OUT)
        gpio.output(17, gpio.LOW)
    else:
        gpio.write(True)
        gpio.write(False)
    
    # Start detection process
    start_time = time.time()
    state_marker.value = 2
    model, model_load_time = load_model(f'final-resources/models/yolov8/{model}', device, gpio, gpio_flag)
    state_marker.value = 1
    state_marker.value = 3
    image_timings = process_images(model, 'final-resources/data/images', results_path, precision, gpio, gpio_flag, 0.5)
    state_marker.value = 1
    total_time = time.time() - start_time
    timings = {
        "model_load_time": model_load_time,
        "image_prediction_times": image_timings,
        "total_execution_time": total_time
    }

    # Save data
    with open(os.path.join(results_path, 'times.json'), 'w') as file:
        json.dump(timings, file, indent=4)
    logging.info(f"END script in {total_time} seconds.")

    # Ending oscilloscope flag
    if gpio_flag:
        gpio.output(17, gpio.LOW)
    else:
        gpio.write(False)

if __name__ == "__main__":
    # Parsing user arguments
    parser = argparse.ArgumentParser(description='Run YOLO Object Detection with specified precision and device.')
    parser.add_argument('precision', type=str,
                        choices=['FP32', 'FP16', 'INT8'],
                        help='Precision of the model (FP32, FP16 or INT8)')
    parser.add_argument('device', type=str,
                        choices=['RPi', 'EdgeTPU', 'RPi-EdgeTPU', 'Rock', 'Rock-EdgeTPU'],
                        help='Device to run the detection on (RPi, EdgeTPU, RPi-EdgeTPU, Rock, Rock-EdgeTPU)')
    args = parser.parse_args()

    # Define paths and model by user args
    results_path, model_path = working_paths(args.precision, args.device)
        
    # Prepare monitoring
    stop_event = threading.Event()
    state_marker = Value('i', 0)  # 'i' working with integer
    monitor_thread = threading.Thread(target=monitor_resources, args=(f'{results_path}/resource_usage.csv', 1, stop_event, state_marker))

    # Prepare logging
    setup_logging(log_path=f'{results_path}/log.txt', log_to_console=False)
    logging.info(f"User device: {args.device}. User precision: {args.precision}")

    try:
        # Start monitoring system resources
        monitor_thread.start()
        # Configure GPIOs by device
        gpio, gpio_flag = set_gpio(args.device)
        # Predict with yolo
        main(args.precision, args.device, results_path, model_path, gpio, gpio_flag, state_marker)
        # End. Clean GPIOs
        if gpio_flag:
            gpio.cleanup() # Clean GPIOs after running
        else:
            gpio.close()

    finally:
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()