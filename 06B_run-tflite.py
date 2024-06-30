"""
Object Detection System using YOLO TFLite models (FP32, FP16 & INT8)

This script configures different Raspberry Pi, Rock and Coral devices to perform object detection on a set of images using the YOLO (You Only Look Once) TFLite model.
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
# from multiprocessing import Value  # Sharing data between processes
import threading  # Thread handling for concurrent tasks
from statistics import mean, stdev # Stadistics
import psutil  # System resource monitoring
import cv2  # Image processing
import numpy as np  # Mathematical operations and multi-dimensional arrays
from tflite_runtime.interpreter import Interpreter  # Interpreter for TFLite models

class GPIOManager:
    """
    Manages GPIO interfaces for various devices, ensuring proper initialization, output control, and cleanup.
    This allows for hardware control signals to be sent, useful for tasks like performance measurement or debugging.
    
    Attributes:
        device (str): The type of device (e.g., 'RPi', 'EdgeTPU').
        pin (int, optional): The GPIO pin number used for signaling. None if GPIO PIN is not used.
        chip (int, optional): The GPIO chip number used for signaling. None if GPIO CHIP is not used.
        gpio (module): The GPIO library module specific to the device.
        gpio_flag (bool): Flag to determine the type of GPIO control used (True for RPi.GPIO, False for periphery).
    """
    def __init__(self, device, pin=None, chip=None):
        self.device = device
        self.pin = pin
        self.chip = chip
        self.gpio = None
        self.gpio_flag = None
        if self.pin is not None:
            self.setup()

    def setup(self):
        """Initializes GPIO based on the device type."""
        if self.device in ['RPi', 'RPi-EdgeTPU']:
            import RPi.GPIO as gpio
            gpio.setmode(gpio.BCM)
            gpio.setup(self.pin, gpio.OUT)
            self.gpio = gpio
            self.gpio_flag = True
        elif self.device in ['EdgeTPU', 'Rock', 'Rock-EdgeTPU']:
            from periphery import GPIO
            self.gpio = GPIO(f"/dev/gpiochip{self.chip}", self.pin, "out")
            self.gpio_flag = False

    def signal_high(self):
        """Activates the GPIO pin signal to high."""
        if self.pin is not None:
            if self.gpio_flag:
                self.gpio.output(self.pin, self.gpio.HIGH)
            else:
                self.gpio.write(True)

    def signal_low(self):
        """Deactivates the GPIO pin signal to low."""
        if self.pin is not None:
            if self.gpio_flag:
                self.gpio.output(self.pin, self.gpio.LOW)
            else:
                self.gpio.write(False)

    def cleanup(self):
        """Cleans up the GPIO settings, freeing the resources."""
        if self.pin is not None:
            if self.gpio_flag:
                self.gpio.cleanup()
            else:
                self.gpio.close()

class ResourceMonitor:
    def __init__(self, interval):
        self.should_continue = threading.Event()
        self.cpu_usages = []
        self.memory_usages = []
        self.interval = interval

    def monitor_resources(self, filename):
        """Function that runs in a thread to continuously monitor resources."""
        if filename is not None:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "CPU (%)", "Memory (%)"])
        
        self.cpu_usages = []  # Reset CPU usage list
        self.memory_usages = []  # Reset memory usage list
        while self.should_continue.is_set():
            cpu, memory = self.get_statistics()
            self.cpu_usages.append(cpu)
            self.memory_usages.append(memory)
            if filename is not None:
                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cpu, memory])
            time.sleep(self.interval)

    def get_statistics(self):
        """Get the current CPU and memory usage."""
        cpu_usage = psutil.cpu_percent(percpu=False)
        memory_usage = psutil.virtual_memory().percent
        return cpu_usage, memory_usage

    def start_monitoring(self, filename):
        """Start the resource monitoring thread."""
        self.should_continue.set()
        self.thread = threading.Thread(target=self.monitor_resources, args=(filename,))
        self.thread.start()

    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        self.should_continue.clear()
        self.thread.join()

    def get_cpu_statistics(self):
        """Return the average and standard deviation of CPU usage."""
        if not self.cpu_usages:
            return None, None
        avg_cpu = mean(self.cpu_usages)
        std_cpu = stdev(self.cpu_usages) if len(self.cpu_usages) > 1 else 0.0
        return avg_cpu, std_cpu

    def get_memory_statistics(self):
        """Return the maximum memory usage."""
        if not self.memory_usages:
            return None
        max_memory = max(self.memory_usages)
        return max_memory

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
        precision (str): Model precision. Can be 'FP32', 'FP16' or 'INT8'.
        device (str): Device to be used. Options are 'Server', 'RPi', 'Rock', 'EdgeTPU', 'RPi-EdgeTPU', and 'Rock-EdgeTPU'.

    Returns:
        tuple: A tuple containing the results path and model path.
    """
    work_path = f'yolov8_{precision}_TFLite'
    if precision == 'FP32' or precision == 'FP16':
        if device == 'RPi':
            work_path += f'_{get_rpi_version()}'
        elif device == 'Rock':
            work_path += '_Rock4Plus'
        elif device == 'Server':
            work_path += '_Server'
        if precision == 'FP32':
            model_path = 'best_float32.tflite'
        else:
            model_path = 'best_float16.tflite'
    
    elif precision == 'INT8' and (device == 'RPi' or device == 'Rock' or device == 'Server'):
        if device == 'RPi':
            work_path += f'_{get_rpi_version()}'
        elif device == 'Rock':
            work_path += '_Rock4Plus'
        elif device == 'Server':
            work_path = '_Server'
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

def load_model(model_path, device, gpio_manager):
    """
    Load a YOLO model from a specified path and measure load time.
    
    Args:
        model_path (str): File path of the YOLO model.
        device (str): Device we are going to use. Can be Server, RPi, Rock, EdgeTPU, RPi-EdgeTPU, Rock-EdgeTPU
        gpio_manager (GPIOManager): Instance of GPIOManager to control GPIO pins. If None, GPIO functions are skipped.

    Returns:
        tuple: (model, load_time) where model is the loaded YOLO model, and load_time is the time taken to load the model in seconds.
    """
    gpio_manager.signal_high()
    start_time = time.time()
    if device == 'RPi' or device == 'Rock' or device == 'Server':
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
    gpio_manager.signal_low()
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time

def process_images(model, imgs_path, results_path, precision, gpio_manager, monitor, bb_conf=0.5, draw=False):
    """
    Process images for object detection and measure processing time.

    Args:
        model: Loaded TFLite model interpreter.
        imgs_path (str): Directory containing images to process.
        results_path (str): Directory containing images results.
        precision (str): Model precision. Can be 'FP32', 'FP16' or 'INT8'.
        gpio_manager (GPIOManager): Instance of GPIOManager to control GPIO pins. If None, GPIO functions are skipped.
        bb_conf (float): Confidence threshold for bounding box predictions.

    Returns:
        list: A list of dictionaries, each containing the filename and the time taken to process that file.
    """

    # Obtain model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    _, model_height, model_width, _ = input_details[0]['shape']
    
    # Work with images
    image_data = {}
    img_counter = 0
    for filename in os.listdir(imgs_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_counter += 1
            image_data[filename] = {
                'Count': img_counter,
                'Predict': [],
                'Time': None
            }
            # Obtain image
            img_path = os.path.join(imgs_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image {filename}")
                continue
            
            # Adapt image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (model_width, model_height))
            if precision == 'FP32' or precision == 'FP16':
                img_norm = img_resized.astype(np.float32) / 255.0
            elif precision == 'INT8':
                img_norm = img_resized.astype(np.int8)
            img_batch = np.expand_dims(img_norm, axis=0)
        
            # Predict
            gpio_manager.signal_high()
            monitor.start_monitoring(None)
            start_time = time.time()
            model.set_tensor(input_details[0]['index'], img_batch)
            model.invoke()
            results = model.get_tensor(output_details[0]['index'])
            end_time = time.time()
            monitor.stop_monitoring()
            avg_cpu_predict, std_cpu_predict = monitor.get_cpu_statistics()
            max_memory_predict = monitor.get_memory_statistics()
            gpio_manager.signal_low()
            image_data[filename]['Time'] = end_time - start_time
            image_data[filename]['Avg CPU'] = avg_cpu_predict
            image_data[filename]['Std dev CPU'] = std_cpu_predict
            image_data[filename]['Max memory'] = max_memory_predict
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
                    image_data[filename]['Predict'].append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                    if draw:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(img, str(round(conf, 1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            
            if draw:
                # Save the image
                save_path = os.path.join(results_path, filename)
                cv2.imwrite(save_path, img)
                logging.info(f"Processed {filename} and saved to {save_path}")

    return image_data

def main(precision, device, save_img, gpio_manager, results_path, model):
    """
    Main function to initialize logging, load the model, and process images.
    
    Args:
        precision (str): Precision of the model ('FP32', 'FP16', 'INT8').
        device (str): Device the model will run on (e.g., 'RPi', 'EdgeTPU').
        gpio_manager (GPIOManager): Instance of GPIOManager to control GPIO pins. If None, GPIO functions are skipped.
        results_path (str): Path where the results will be saved.
        model_path (str): Path to the TFLite model file.
        state_marker (multiprocessing.Value): Shared variable used to track the state of the process. It's used to communicate between threads or processes.
            - 0: "Low" state, indicating no activity.
            - 1: "High" state, indicating the model is loaded.
            - 2: "High" state, indicating detection process is ongoing.
    """

    monitor = ResourceMonitor(0.1)
   
    # Starting oscilloscope flag
    gpio_manager.signal_high()
    gpio_manager.signal_low()
    
    # Load model
    start_time = time.time()
    monitor.start_monitoring(None)
    model, model_load_time = load_model(f'final-resources/models/yolov8/{model}', device, gpio_manager)
    monitor.stop_monitoring()
    avg_cpu_load_model, std_cpu_load_model = monitor.get_cpu_statistics()
    max_memory_load_model = monitor.get_memory_statistics()
    # Predict
    results = process_images(model, 'final-resources/data/images', results_path, precision, gpio_manager, monitor, 0.5, save_img)
    total_time = time.time() - start_time
    
    # Save data
    results['Load model'] = {'Avg CPU': avg_cpu_load_model, 'Std dev CPU':std_cpu_load_model, 'Max memory': max_memory_load_model, 'Load time': model_load_time}
    results['total execution time'] = total_time
    name = results_path.split('/')[-1]
    with open(os.path.join(results_path, f'{name}_results.json'), 'w') as file:
        json.dump(results, file, indent=4)
    logging.info(f"END script in {total_time} seconds.")

    # Ending oscilloscope flag
    gpio_manager.signal_high()
    gpio_manager.signal_low()

    # Clean GPIOs
    gpio_manager.cleanup()

if __name__ == "__main__":
    # Parsing user arguments
    parser = argparse.ArgumentParser(description='Run YOLO Object Detection with specified precision and device.')
    parser.add_argument('precision', type=str,
                        choices=['FP32', 'FP16', 'INT8'],
                        help='Precision of the model (FP32, FP16 or INT8)')
    parser.add_argument('device', type=str,
                        choices=['Server', 'RPi', 'EdgeTPU', 'RPi-EdgeTPU', 'Rock', 'Rock-EdgeTPU'],
                        help='Device to run the detection on (Server, RPi, EdgeTPU, RPi-EdgeTPU, Rock, Rock-EdgeTPU)')
    parser.add_argument('--gpio_pin', type=int, default=None,
                        help='Optional GPIO pin number to use for signaling. If not provided, GPIO PIN is not used.')
    parser.add_argument('--gpio_chip', type=int, default=None,
                        help='Optional GPIO chip number to use for signaling. If not provided, GPIO CHIP is not used.')
    parser.add_argument('--save_img', type=bool, default=False,
                        help='Optional flag to save the image results. If not provided, False.')
    args = parser.parse_args()

    # Define paths and model by user args
    results_path, model_path = working_paths(args.precision, args.device)
        
    # Prepare monitoring
    # stop_event = threading.Event()
    # state_marker = Value('i', 0)  # 'i' working with integer
    # monitor_thread = threading.Thread(target=monitor_resources, args=(f'{results_path}/resource_usage.csv', 1, stop_event, state_marker))

    # Prepare logging
    setup_logging(log_path=f'{results_path}/log.txt', log_to_console=False)
    logging.info(f"User device: {args.device}. User precision: {args.precision}. Using GPIO pin: {args.gpio_pin}")

    try:
        # Start monitoring system resources
        # monitor_thread.start()
        # Configure GPIOs by device
        gpio_manager = GPIOManager(device=args.device, pin=args.gpio_pin, chip=args.gpio_chip)
        # Predict with yolo
        main(args.precision, args.device, args.save_img, gpio_manager, results_path, model_path)

    finally:
        # Stop monitoring
        # stop_event.set()
        # monitor_thread.join()
        pass