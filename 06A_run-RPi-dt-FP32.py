"""
Object Detection System using YOLO Ultralytics FP32 model and Raspberry Pi

This script configures a Raspberry Pi to perform object detection on a set of images using the YOLO (You Only Look Once) model.
It includes functions for setting up logging, loading the YOLO model, processing images to detect objects,
and saving timing data for analysis. GPIO pins are used to signal the start and end of significant operations,
which can be monitored with an oscilloscope for debugging or performance measurement.
"""

import cv2
import os
import time
import json
import logging
from ultralytics import YOLO 
import RPi.GPIO as GPIO      # GPIO for Raspberry Pi interaction.

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

def load_model(model_path):
    """
    Load a YOLO model from a specified path and measure load time.
    
    Args:
        model_path (str): File path of the YOLO model.

    Returns:
        tuple: (model, load_time) where model is the loaded YOLO model, and load_time is the time taken to load the model in seconds.
    """
    GPIO.output(17, GPIO.HIGH)  # Signal GPIO pin before loading model.
    start_time = time.time()
    model = YOLO(model_path)
    end_time = time.time()
    GPIO.output(17, GPIO.LOW)   # Signal GPIO pin after loading model.
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time

def process_images(model, imgs_path, results_path, bb_conf=0.5):
    """
    Process images for object detection and measure processing time.
    
    Args:
        model: Loaded YOLO model.
        imgs_path (str): Directory containing images to process.
        results_path (str): Directory containing images results.
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
            GPIO.output(17, GPIO.HIGH)  # Signal GPIO pin before prediction.
            start_time = time.time()
            results = model.predict(img_path)[0]
            end_time = time.time()
            GPIO.output(17, GPIO.LOW)   # Signal GPIO pin after prediction.
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

def main(work_path, model):
    """
    Main function to initialize logging, load the model, and process images.
    
    Args:
        work_path (str): Working directory path where results and logs will be saved.
        model (str): Model filename to be loaded.
    """
    
    results_path = f'results/{work_path}'
    
    # Starting oscilloscope flag
    GPIO.setup(17, GPIO.OUT)
    GPIO.output(17, GPIO.LOW)
    
    setup_logging(log_path=f'{results_path}/log.txt', log_to_console=False)
    
    # Start detection process
    start_time = time.time()
    model, model_load_time = load_model(f'final-resources/models/yolov8/{model}')
    image_timings = process_images(model, 'final-resources/data/images', results_path, 0.5)
    total_time = time.time() - start_time
    timings = {
        "model_load_time": model_load_time,
        "image_prediction_times": image_timings,
        "total_execution_time": total_time
    }

    # Save data
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, 'times.json'), 'w') as file:
        json.dump(timings, file, indent=4)
    logging.info(f"END script in {total_time} seconds.")

    # Ending oscilloscope flag
    GPIO.setup(17, GPIO.OUT)
    GPIO.output(17, GPIO.LOW)

if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM) # Setup GPIO mode
    GPIO.cleanup() # Ensure GPIOs are ok
    main('yolov8_FP32_pt_RPi', 'best.pt')
    GPIO.cleanup() # Clean GPIOs
