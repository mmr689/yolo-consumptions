
"""
Código RPi antiguo adaptado alo nuevo
"""

from tflite_runtime.interpreter import Interpreter
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.LOW)

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


directory = 'final-resources/data/images'
for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            frame = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



            # Ruta al modelo TFLite
            GPIO.output(17, GPIO.HIGH)
            model_path = os.path.join('final-resources/models/yolov8/','best_float32.tflite')
            interpreter = Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            print('INTERPRETER')
            print(interpreter)
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            GPIO.output(17, GPIO.LOW)

            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            _, height, width, _ = input_details[0]['shape']
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            print('input')
            print(input_details)
            print('output')
            print(output_details)
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')


            frame_resized = cv2.resize(frame_rgb, (width, height))
            # Normalizar los valores de píxeles a FLOAT32
            input_data = frame_resized.astype(np.float32) / 255.0
            # Agregar una dimensión para representar el lote (batch)
            input_data = np.expand_dims(input_data, axis=0)
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            print('input_data')
            print(input_data)
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')


            # Perform the actual detection by running the model with the image as input
            GPIO.output(17, GPIO.HIGH)
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            print('output_data')
            print(output_data)
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            GPIO.output(17, GPIO.LOW)


            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')
            print('output_details[0][shape][2]')
            print(output_details[0]['shape'][2])
            print(' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

            # 
            bb_dict = {}
            for i in range(output_details[0]['shape'][2]):
                probs = output_data[0][4:, i].flatten() # CONF LABELS
                if np.max(probs) > 0.25:
                    x, y, w, h = output_data[0][:4, i].flatten() # COORDS
                    print(i, np.max(probs), np.argmax(probs), (x, y, w, h))
                    # print(i, np.max(probs), np.argmax(probs), (x, y, w, h))

                    # Coordenadas del punto (ejemplo)
                    x = int(x * frame.shape[1])
                    y = int(y * frame.shape[0])

                    # Dimensiones del rectángulo
                    width = int(w * frame.shape[1])
                    height = int(h * frame.shape[0])

                    # Calcular las coordenadas del vértice superior izquierdo del rectángulo
                    x_izquierda = x - width // 2
                    y_arriba = y - height // 2
                        
                    # Guardar
                    if np.argmax(probs) not in bb_dict:
                        bb_dict[np.argmax(probs)] = [(x_izquierda, y_arriba, x_izquierda + width, y_arriba + height, np.max(probs))]
                    else:
                        bb_dict[np.argmax(probs)].append((x_izquierda, y_arriba, x_izquierda + width, y_arriba + height, np.max(probs)))

            # Aplicamos NMS
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
                    print(key, x1, y1, x2, y2, conf)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, str(round(conf, 1)), (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3, cv2.LINE_AA)

            # Guardar la imagen resultante con rectángulos dibujados
            output_path = 'results/yolov8_FP32_TFLite/'+filename
            cv2.imwrite(output_path, frame)
GPIO.cleanup()
