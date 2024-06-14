"""
Object Detection System using YOLO TFLite FP32 model and Raspberry Pi

This script configures a Raspberry Pi to perform object detection on a set of images using the YOLO (You Only Look Once) TFLite FP32 model.
It includes functions for setting up logging, loading the YOLO TFLite model, processing images to detect objects,
and saving timing data for analysis. GPIO pins are used to signal the start and end of significant operations,
which can be monitored with an oscilloscope for debugging or performance measurement.
"""

from tflite_runtime.interpreter import Interpreter
import os
import time
import json
import logging
import cv2
import numpy as np
import RPi.GPIO as GPIO

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

def load_model(model_path, device):
    """
    Load a YOLO model from a specified path and measure load time.
    
    Args:
        model_path (str): File path of the YOLO model.

    Returns:
        tuple: (model, load_time) where model is the loaded YOLO model, and load_time is the time taken to load the model in seconds.
    """
    GPIO.output(17, GPIO.HIGH) # Signal GPIO pin before loading model.
    start_time = time.time()
    if device == 'RPi':
        model = Interpreter(model_path=model_path)
    elif device == 'EdgeTPU':
        from tflite_runtime.interpreter import load_delegate
        model = Interpreter(model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'pci:0'})])
    elif device == 'USB-EdgeTPU':
        from tflite_runtime.interpreter import load_delegate
        model = Interpreter(model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1', options={'device': 'usb'})])
    model.allocate_tensors()
    end_time = time.time()
    GPIO.output(17, GPIO.LOW) # Signal GPIO pin after loading model.
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time

def process_images(model, imgs_path, results_path, precision, bb_conf=0.5):
    """
    Process images for object detection and measure processing time.

    Args:
        model: Loaded TFLite model interpreter.
        imgs_path (str): Directory containing images to process.
        results_path (str): Directory containing images results.
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
            if precision == 'FP32':
                img_norm = img_resized.astype(np.float32) / 255.0
            elif precision == 'INT8':
                img_norm = img_resized.astype(np.int8)
            img_batch = np.expand_dims(img_norm, axis=0)
            
            # Predict
            GPIO.output(17, GPIO.HIGH) # Signal GPIO pin before prediction.
            start_time = time.time()
            model.set_tensor(input_details[0]['index'], img_batch)
            model.invoke()
            results = model.get_tensor(output_details[0]['index'])
            end_time = time.time()
            GPIO.output(17, GPIO.LOW) # Signal GPIO pin after prediction.
            image_timings.append({filename: end_time - start_time})
            # Obtain all bounding boxes and confidences
            bb_dict = {}
            for i in range(output_details[0]['shape'][2]):
                confs = results[0][4:, i].flatten()
                if precision == 'FP32':
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

def main(precision, device):
    """
    Main function to initialize logging, load the model, and process images.
    
    Args:
        precision (str): Model precision. Can be FP32 or INT8.
        device (str): Device we are going to use. Can be RPi, EdgeTPU, USB-EdgeTPU
    """
    
    # Starting oscilloscope flag
    GPIO.setup(17, GPIO.OUT)
    GPIO.output(17, GPIO.LOW)
    
    work_path = f'yolov8_{precision}_TFLite'
    if precision == 'FP32':
        work_path += '_RPi'
        model = 'best_float32.tflite'
    elif precision == 'INT8' and device == 'RPi':
        work_path += '_RPi'
        model = 'best_full_integer_quant.tflite'
    elif precision == 'INT8' and 'EdgeTPU' in device:
        if 'USB' in device:
            work_path += '_RPi-USBCoral'
        elif 'mini' in device:
            work_path += '_DevBoardMini'
        else:
            work_path += '_DevBoard'
        model = 'best_full_integer_quant_edgetpu.tflite'
    results_path = f'results/{work_path}'
    if not os.path.exists(results_path):
        print(f"Directory {results_path} does not exist. Please create it to proceed.")

    setup_logging(log_path=f'{results_path}/log.txt', log_to_console=False)
    logging.info(f"User device: {device}. User precision: {precision}")
    
    # Start detection process
    start_time = time.time()
    model, model_load_time = load_model(f'final-resources/models/yolov8/{model}', device)
    image_timings = process_images(model, 'final-resources/data/images', results_path, precision, 0.5)
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
    main('INT8', 'USB-EdgeTPU')
    GPIO.cleanup() # Clean GPIOs
