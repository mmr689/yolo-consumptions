
"""
Código RPi antiguo adaptado alo nuevo
"""

from tflite_runtime.interpreter import Interpreter
import os
import time
import json
import logging
import cv2
import numpy as np
import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(17, GPIO.OUT)
# GPIO.output(17, GPIO.LOW)

def calcular_solapamiento(rect1, rect2):
    x1_1, y1_1, x2_1, y2_1, confianza1 = rect1
    x1_2, y1_2, x2_2, y2_2, confianza2 = rect2

    # Calcular áreas de los rectángulos
    area_rect1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_rect2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calcular intersección
    interseccion_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    interseccion_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    area_interseccion = interseccion_x * interseccion_y

    # Calcular solapamiento como fracción del área más pequeña
    solapamiento = area_interseccion / min(area_rect1, area_rect2)

    return solapamiento

def eliminar_solapamientos(lista_rectangulos):
    rectangulos_eliminados = []
    i = 0
    while i < len(lista_rectangulos):
        j = i + 1
        while j < len(lista_rectangulos):
            if calcular_solapamiento(lista_rectangulos[i], lista_rectangulos[j]) > 0.9:
                # Almacenar en la lista de rectángulos eliminados
                rectangulos_eliminados.append(lista_rectangulos[j])
                del lista_rectangulos[j]
            else:
                j += 1
        i += 1
    return rectangulos_eliminados

# directory = 'final-resources/data/images'
# for filename in os.listdir(directory):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(directory, filename)
#             frame = cv2.imread(img_path)
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



#             # Ruta al modelo TFLite
#             GPIO.output(17, GPIO.HIGH)
#             model_path = os.path.join('final-resources/models/yolov8/','best_float32.tflite')
#             interpreter = Interpreter(model_path=model_path)
#             interpreter.allocate_tensors()
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             print('INTERPRETER')
#             print(interpreter)
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             GPIO.output(17, GPIO.LOW)

#             # Get model details
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()
#             _, height, width, _ = input_details[0]['shape']
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             print('input')
#             print(input_details)
#             print('output')
#             print(output_details)
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')


#             frame_resized = cv2.resize(frame_rgb, (width, height))
#             # Normalizar los valores de píxeles a FLOAT32
#             input_data = frame_resized.astype(np.float32) / 255.0
#             # Agregar una dimensión para representar el lote (batch)
#             input_data = np.expand_dims(input_data, axis=0)
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             print('input_data')
#             print(input_data)
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')


#             # Perform the actual detection by running the model with the image as input
#             GPIO.output(17, GPIO.HIGH)
#             interpreter.set_tensor(input_details[0]['index'],input_data)
#             interpreter.invoke()
#             output_data = interpreter.get_tensor(output_details[0]['index'])
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             print('output_data')
#             print(output_data)
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             GPIO.output(17, GPIO.LOW)


#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
#             print('output_details[0][shape][2]')
#             print(output_details[0]['shape'][2])
#             print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

#             # 
#             bb_dict = {}
#             for i in range(output_details[0]['shape'][2]):
#                 probs = output_data[0][4:, i].flatten() # CONF LABELS
#                 if np.max(probs) > 0.5:
#                     x, y, w, h = output_data[0][:4, i].flatten() # COORDS
#                     print(i, np.max(probs), np.argmax(probs), (x, y, w, h))
#                     # print(i, np.max(probs), np.argmax(probs), (x, y, w, h))

#                     # Coordenadas del punto (ejemplo)
#                     x = int(x * frame.shape[1])
#                     y = int(y * frame.shape[0])

#                     # Dimensiones del rectángulo
#                     width = int(w * frame.shape[1])
#                     height = int(h * frame.shape[0])

#                     # Calcular las coordenadas del vértice superior izquierdo del rectángulo
#                     x_izquierda = x - width // 2
#                     y_arriba = y - height // 2
                        
#                     # Guardar
#                     if np.argmax(probs) not in bb_dict:
#                         bb_dict[np.argmax(probs)] = [(x_izquierda, y_arriba, x_izquierda + width, y_arriba + height, np.max(probs))]
#                     else:
#                         bb_dict[np.argmax(probs)].append((x_izquierda, y_arriba, x_izquierda + width, y_arriba + height, np.max(probs)))

#             # Aplicamos NMS
#             rectangulos_eliminados = []
#             for key,vals in bb_dict.items():
#                 # Ordenar la lista por el quinto valor de las tuplas (confianza) de manera descendente
#                 vals = sorted(vals, key=lambda x: x[4], reverse=True)
#                 # Eliminar solapamientos mientras haya
#                 while True:
#                     cantidad_anterior = len(vals)
#                     rectangulos_eliminados.extend(eliminar_solapamientos(vals))
#                     cantidad_actual = len(vals)

#                     # Salir del bucle si no hay cambios
#                     if cantidad_anterior == cantidad_actual:
#                         break

#                 # Mostrar resultado
#                 for rectangulo in vals:
                    
#                     x1, y1, x2, y2, conf = rectangulo
#                     print(key, x1, y1, x2, y2, conf)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                     cv2.putText(frame, str(round(conf, 1)), (int(x1), int(y1)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3, cv2.LINE_AA)

#             # Guardar la imagen resultante con rectángulos dibujados
#             output_path = 'results/yolov8_FP32_TFLite/'+filename
#             cv2.imwrite(output_path, frame)
# GPIO.cleanup()


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
    model = Interpreter(model_path=model_path)
    model.allocate_tensors()
    end_time = time.time()
    GPIO.output(17, GPIO.LOW)
    logging.info(f"Model loaded in {end_time - start_time} seconds.")
    return model, end_time - start_time


def process_images(model, directory, bb_conf=0.5):
    """Process images and return timings."""

    # Get model details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    _, model_height, model_width, _ = input_details[0]['shape']

    image_timings = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image {filename}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (model_width, model_height))
            img_norm = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_norm, axis=0)

            # Detect
            GPIO.output(17, GPIO.HIGH)
            start_time = time.time()
            model.set_tensor(input_details[0]['index'], img_batch)
            model.invoke()
            results = model.get_tensor(output_details[0]['index'])
            end_time = time.time()
            GPIO.output(17, GPIO.LOW)
            image_timings.append({filename: end_time - start_time})


            GPIO.output(17, GPIO.HIGH)
            # Obtener bounding boxes y scores - HACER FUNCIÓN AQUÍ
            bb_dict = {}
            for i in range(output_details[0]['shape'][2]):
                confs = results[0][4:, i].flatten() # CONF LABELS
                if np.max(confs) > bb_conf:
                    x, y, w, h = results[0][:4, i].flatten() # COORDS

                    # Coordenadas del punto
                    x, y = int(x * img.shape[1]), int(y * img.shape[0]) 
                    width, height = int(w * img.shape[1]), int(h * img.shape[0])

                    # Calcular las coordenadas de las bb
                    x1, y1 = x-width//2, y-height//2
                    x2, y2 ,conf = x1+width, y1+height, np.max(confs)
                    
                    # Guardar para cada etiquetas las bb y conf
                    if np.argmax(confs) not in bb_dict:
                        bb_dict[np.argmax(confs)] = [(x1, y1, x2, y2, conf)]
                    else:
                        bb_dict[np.argmax(confs)].append((x1, y1, x2, y2, conf))

            # Aplicamos NMS - HACER FUNCIÓN AQUÍ
            rectangulos_eliminados = []
            for key,vals in bb_dict.items():
                # Ordenar la lista por el quinto valor de las tuplas (confianza) de manera descendente
                vals = sorted(vals, key=lambda x: x[4], reverse=True)
                # Eliminar solapamientos mientras haya
                while True:
                    cantidad_anterior = len(vals)
                    rectangulos_eliminados.extend(eliminar_solapamientos(vals))
                    cantidad_actual = len(vals)

                    # Salir del bucle si no hay cambios
                    if cantidad_anterior == cantidad_actual:
                        break

                # Mostrar resultado
                for rectangulo in vals:
                    x1, y1, x2, y2, conf = rectangulo
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, str(round(conf, 1)), (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3, cv2.LINE_AA)
            GPIO.output(17, GPIO.LOW)


        save_path = os.path.join('results/yolov8_FP32_TFLite', filename)
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
    model, model_load_time = load_model('final-resources/models/yolov8/best_float32.tflite')
    image_timings = process_images(model, 'final-resources/data/images', 0.5)
    total_time = time.time() - start_time
    timings = {
        "model_load_time": model_load_time,
        "image_prediction_times": image_timings,
        "total_execution_time": total_time
    }
    print(timings)

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