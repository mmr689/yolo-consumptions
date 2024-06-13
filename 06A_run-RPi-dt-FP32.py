
import cv2
import os
import time
import json
import logging
from ultralytics import YOLO
import RPi.GPIO as GPIO

def setup_logging( log_path, log_to_console=True,):
    """Setup logging configuration with an option to log to console."""
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
    """Load and return the YOLO model."""
    GPIO.output(17, GPIO.HIGH)
    start_time = time.time()
    model = YOLO(model_path)
    end_time = time.time()
    GPIO.output(17, GPIO.LOW)
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time

def process_images(model, directory, bb_conf=0.5):
    """Process images and return timings."""
    image_timings = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image {filename}")
                continue

            GPIO.output(17, GPIO.HIGH)
            start_time = time.time()
            results = model.predict(img_path)[0]
            end_time = time.time()
            GPIO.output(17, GPIO.LOW)
            image_timings.append({filename: end_time - start_time})

            GPIO.output(17, GPIO.HIGH)
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = result
                if conf >= bb_conf:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(img, str(round(conf,1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            GPIO.output(17, GPIO.LOW)

            save_path = os.path.join('results/yolov8_FP32_pt', filename)
            cv2.imwrite(save_path, img)
            logging.info(f"Processed {filename} and saved to {save_path}")
    return image_timings

def main():
    # Creamos un pico con los GPIOs para observarlo en el osciloscopio
    GPIO.setup(17, GPIO.OUT)
    GPIO.output(17, GPIO.LOW)
    
    # Configuramos logs
    setup_logging(log_path='results/yolov8_FP32_pt/log.txt', log_to_console=False)
    
    # Cargamos el modelo y predecimos
    start_time = time.time()
    model, model_load_time = load_model('final-resources/models/yolov8/best.pt')
    image_timings = process_images(model, 'final-resources/data/images', 0.5)
    total_time = time.time() - start_time
    timings = {
        "model_load_time": model_load_time,
        "image_prediction_times": image_timings,
        "total_execution_time": total_time
    }

    # Save timing data
    results_path = 'results/yolov8_FP32_pt'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, 'times.json'), 'w') as file:
        json.dump(timings, file, indent=4)
    logging.info("Execution times and logs have been saved.")

    # Creamos un pico con los GPIOs para observarlo en el osciloscopio
    GPIO.setup(17, GPIO.OUT)
    GPIO.output(17, GPIO.LOW)

if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
    main()
    GPIO.cleanup()