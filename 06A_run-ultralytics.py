"""
Object Detection System using YOLO Ultralytics FP32 model

This script configures Raspberry Pi and Rock boards to perform object detection on a set of images using the YOLO (You Only Look Once) model.
It includes functions for setting up logging, monitoring resources, loading the YOLO model, processing images to detect objects,
and saving timing data for analysis. GPIO pins are used to signal the start and end of significant operations,
which can be monitored with an oscilloscope for debugging or performance measurement.
"""

import argparse
import cv2
import os
import time
import json
import logging
from ultralytics import YOLO

import threading
from multiprocessing import Value
import psutil
import csv
import time
from datetime import datetime


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

def setup_logging(log_path, log_to_console=True):
    """
    Set up logging configuration.
    
    Args:
        log_path (str): Path to save the log file.
        log_to_console (bool): If True, logs will also be printed to the console.
    """
    # Create directory for log if it doesn't exist.
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

def load_model(model_path, gpio):
    """
    Load a YOLO model from a specified path and measure load time.
    
    Args:
        model_path (str): File path of the YOLO model.
        gpio: GPIO interface for controlling hardware signals. Can be None or a GPIO object.

    Returns:
        tuple: (model, load_time) where model is the loaded YOLO model, and load_time is the time taken to load the model in seconds.
    """
    if gpio is None:
        GPIO.output(17, GPIO.HIGH) # Signal GPIO pin before loading model.
    else:
        gpio.write(True)
    start_time = time.time()
    model = YOLO(model_path)
    end_time = time.time()
    if gpio is None:
        GPIO.output(17, GPIO.LOW) # Signal GPIO pin after loading model.
    else:
        gpio.write(False)
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time

def process_images(model, imgs_path, results_path, gpio, bb_conf=0.5):
    """
    Process images for object detection and measure processing time.
    
    Args:
        model: Loaded YOLO model.
        imgs_path (str): Directory containing images to process.
        results_path (str): Directory containing images results.
        gpio: GPIO interface for controlling hardware signals. Can be None or a GPIO object.
        bb_conf (float): Confidence threshold for bounding box predictions.

    Returns:
        list: A list of dictionaries, each containing the filename and the time taken to process that file.
    """
    image_timings = []
    for filename in os.listdir(imgs_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Obtain image
            img_path = os.path.join(imgs_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image {filename}")
                continue
            
            # Obtain image
            if gpio is None:
                GPIO.output(17, GPIO.HIGH) # Signal GPIO pin before loading model.
            else:
                gpio.write(True)
            start_time = time.time()
            results = model.predict(img_path)[0]
            end_time = time.time()
            if gpio is None:
                GPIO.output(17, GPIO.LOW) # Signal GPIO pin after loading model.
            else:
                gpio.write(False)
            image_timings.append({filename: end_time - start_time})

            # Draw bounding boxes
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = result
                if conf >= bb_conf:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(img, str(round(conf,1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

            # Save the images
            save_path = os.path.join(results_path, filename)
            cv2.imwrite(save_path, img)
            logging.info(f"Processed {filename} and saved to {save_path}")
    return image_timings

def main(device, results_path, model, gpio, state_marker):
    """
    Main function to initialize logging, load the model, and process images.
    
    Args:
        device (str): Device to be used for running the model. Options are 'RPi', 'Rock', 'EdgeTPU', 'RPi-EdgeTPU', and 'Rock-EdgeTPU'.
        results_path (str): Path to save the results and timings.
        model (str): Path to the model to be loaded.
        gpio: GPIO interface for controlling hardware signals. Can be None or a GPIO object.
        state_marker (multiprocessing.Value): Shared integer used for tracking the state of the process.
            - 0: Initial state.
            - 1: Model loaded.
            - 2: Detection process started.
            - 3: Image processing started.
    """
    
    # Starting oscilloscope flag
    state_marker.value = 0
    if gpio is None:
        GPIO.setup(17, GPIO.OUT)
        GPIO.output(17, GPIO.LOW)
    else:
        gpio.write(True)
        gpio.write(False)
    
    # Start detection process
    start_time = time.time()
    state_marker.value = 2
    model, model_load_time = load_model(f'final-resources/models/yolov8/{model}', gpio)
    state_marker.value = 1
    state_marker.value = 3
    image_timings = process_images(model, 'final-resources/data/images', results_path, gpio, 0.5)
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
    if gpio is None:
        GPIO.output(17, GPIO.LOW)
    else:
        gpio.write(False)

if __name__ == "__main__":
    # Parsing user arguments
    parser = argparse.ArgumentParser(description='Run YOLO Object Detection with specified device.')
    parser.add_argument('device', type=str,
                        choices=['RPi', 'Rock'],
                        help='Device to run the detection on (RPi, Rock)')
    args = parser.parse_args()

    # Define paths and model by user args
    work_path = f'yolov8_FP32_Ultralytics'
    if args.device == 'RPi':
        work_path += '_RPi'
    elif args.device == 'Rock':
        work_path += '_Rock4Plus'
    results_path = f'results/{work_path}'
    model = 'best.pt'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Prepare monitoring
    stop_event = threading.Event()
    state_marker = Value('i', 0)  # 'i' working with integer
    monitor_thread = threading.Thread(target=monitor_resources, args=(f'{results_path}/resource_usage.csv', 1, stop_event, state_marker))
    
    # Prepare logging
    setup_logging(log_path=f'{results_path}/log.txt', log_to_console=False)
    logging.info(f"User device: {args.device}. Precision: FP32")

    try:
        monitor_thread.start()
    
        if args.device == 'RPi':
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM) # Setup GPIO mode
            gpio = None
            
        elif args.device == 'Rock':
            from periphery import GPIO
            gpio = GPIO("/dev/gpiochip3", 15, "out")
            

        main(args.device, results_path, model, gpio, state_marker)

        if gpio is None:
            GPIO.cleanup() # Clean GPIOs after running
        else:
            gpio.close()

    finally:
        stop_event.set()
        monitor_thread.join()