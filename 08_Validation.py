"""
Código de detección sin control de parámetros cómo tiempo, cpu...
"""

import os  # OS operations
import cv2  # Image processing
import numpy as np  # Mathematical operations and multi-dimensional arrays
import pandas as pd
from tflite_runtime.interpreter import Interpreter  # Interpreter for TFLite models

def preprocess_image(img, model_width, model_height, precision, convert_rgb=True):
    """
    Convert and resize the image for model prediction.

    Args:
        img (numpy.ndarray): The original image.
        model_width (int): The required width of the image for the model.
        model_height (int): The required height of the image for the model.
        precision (str): The precision mode of the model, affects how image data is normalized.
        convert_rgb (bool): If True, convert image color from BGR to RGB. Default is True.

    Returns:
        numpy.ndarray: The preprocessed image ready for model input.
    """
    # Convert color from BGR to RGB if required
    if convert_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the required dimensions
    img_resized = cv2.resize(img, (model_width, model_height))
    
    # Normalize the image based on the precision specified
    if precision in ['FP32', 'FP16']:
        img_norm = img_resized.astype(np.float32) / 255.0
    elif precision == 'INT8':
        img_norm = img_resized.astype(np.int8)
    
    # Add a batch dimension
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_batch

def convert_xywh_to_xyxy(x, y, w, h, image_width, image_height):
    """
    Convert normalized bbox coordinates from (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        x (float): Normalized x coordinate of the bbox center (0 to 1).
        y (float): Normalized y coordinate of the bbox center (0 to 1).
        w (float): Normalized width of the bbox (0 to 1).
        h (float): Normalized height of the bbox (0 to 1).
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        tuple: A tuple of integers (x1, y1, x2, y2), representing the top-left and bottom-right corners of the bbox.
    """
    # Convert from normalized to absolute pixel coordinates
    center_x = int(x * image_width)
    center_y = int(y * image_height)
    width = int(w * image_width)
    height = int(h * image_height)

    # Calculate the top-left corner
    x1 = center_x - width // 2
    y1 = center_y - height // 2

    # Calculate the bottom-right corner
    x2 = x1 + width
    y2 = y1 + height

    return (x1, y1, x2, y2)

def extract_bounding_boxes(bb_conf, model, img, results):
    """
    Extract and filter bounding boxes from model detections based on confidence threshold.

    Args:
        bb_conf (float): Confidence threshold for bounding box detections.
        model (TFLiteModel): An instance of the TFLiteModel class.
        img (np.array): The image being processed.
        results (np.array): The raw output from the model prediction.

    Returns:
        dict: A dictionary with labels as keys and lists of bounding box coordinates and confidences as values.
    """
    bb_dict = {}
    for i in range(model.max_detections):
        # Obtain coords (xywh) and confidence of detections
        coords, conf, label = model.extract_detections(results, i)
        # Save coordinates of bounding boxes if confidence exceeds threshold
        if conf >= bb_conf:
            x, y, w, h = coords
            x1, y1, x2, y2 = convert_xywh_to_xyxy(x, y, w, h, img.shape[1], img.shape[0])
                     
            if label not in bb_dict:
                bb_dict[label] = [(x1, y1, x2, y2, conf)]
            else:
                bb_dict[label].append((x1, y1, x2, y2, conf))
    return bb_dict

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

def calculate_overlap(rect1, rect2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        rect1, rect2 (tuple): Bounding boxes specified as (x1, y1, x2, y2, confidence).

    Returns:
        float: IoU value.
    """
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union_area = rect1_area + rect2_area - intersection_area

    return intersection_area / union_area

def apply_nms(bb_dict, overlap_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes based on the Intersection over Union (IoU) metric.
    
    Args:
        bb_dict (dict of lists): Dictionary with labels as keys and lists of bounding boxes as values. Each bounding box is represented as a tuple (x1, y1, x2, y2, confidence).
        overlap_threshold (float): IoU threshold for determining whether to suppress a bounding box. Boxes with IoU above this threshold will be considered for suppression.

    Returns:
        dict of lists: Filtered dictionary with bounding boxes that have been retained after applying NMS.
    """
    nms_dict = {}
    for label, boxes in bb_dict.items():
        # Keep boxes after applying NMS
        kept_boxes = []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence
        
        while boxes:
            current_box = boxes.pop(0)
            kept_boxes.append(current_box)
            # Only keep boxes that do not have significant overlap
            boxes = [box for box in boxes if calculate_overlap(current_box, box) < overlap_threshold]

        nms_dict[label] = kept_boxes

    return nms_dict

def draw_bounding_boxes_on_image(img, bb_dict):
    """
    Draw bounding boxes on the image based on the bounding box dictionary.

    Args:
        img (np.array): Image on which to draw the bounding boxes.
        bb_dict (dict): Dictionary containing labels and bounding boxes.

    Returns:
        np.array: The image with bounding boxes drawn.
    """
    for _, boxes in bb_dict.items():
        for x1, y1, x2, y2, conf in boxes:
            # Draw rectangle on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optionally add label and confidence score
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

class TFLiteModel:
    """
    A class to handle loading and inference with a TensorFlow Lite model.

    Attributes:
        model_path (str): The file path to the TFLite model.
        model (Interpreter): An instance of the TFLite Interpreter for model operations.
        input_details (list): Details about the input tensors for the model.
        output_details (list): Details about the output tensors for the model.
        width (int): The expected width of the input image for the model.
        height (int): The expected height of the input image for the model.
        max_detections (int): The maximum number of detections the model can output.
        precision (str): The precision type of the model ('FP32', 'FP16', 'INT8').
    """

    def __init__(self, model_path, precision='FP32'):
        """
        Initializes the TFLiteModel with the specified model.

        Args:
            model_path (str): The file path to the TFLite model.
        """
        self.model_path = model_path
        self.precision = precision
        self.model = self.load_model()
        # Retrieve details about the model, such as input/output tensors and model dimensions.
        self.input_details, self.output_details, self.width, self.height, self.max_detections = self.get_model_details()

    def load_model(self):
        """
        Loads the TFLite model and allocates tensors.

        Returns:
            Interpreter: An instance of the TFLite Interpreter with allocated tensors.
        """
        model = Interpreter(model_path=self.model_path)
        model.allocate_tensors()
        return model

    def get_model_details(self):
        """
        Retrieves and stores details about the model's input and output tensors,
        and the dimensions required for input images.

        Returns:
            tuple: A tuple containing input details, output details, input width,
            input height, and the maximum number of detections.
        """
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        _, height, width, _ = input_details[0]['shape']
        max_detections = output_details[0]['shape'][2]  # Typically the number of possible bounding boxes.
        return input_details, output_details, width, height, max_detections

    def predict(self, img_batch):
        """
        Runs prediction on the processed image batch.

        Args:
            img_batch (np.array): A batch of preprocessed images ready for model inference.

        Returns:
            np.array: The output tensor containing the prediction results.
        """
        self.model.set_tensor(self.input_details[0]['index'], img_batch)
        self.model.invoke()
        results = self.model.get_tensor(self.output_details[0]['index'])
        return results
    
    def extract_detections(self, results, index):
        """
        Extract bounding box coordinates and class confidences from the model results,
        adjusting for the specified model precision.
        """
        # Adjust coordinates for INT8 precision
        if self.precision == 'INT8':
            coords = ((results[0][:4, index].flatten() + 128) / 255).tolist()
        else:
            coords = results[0][:4, index].flatten()

        # Adjust confidences for INT8 precision
        if self.precision == 'INT8':
            confs = ((results[0][4:, index].flatten() + 128) / 255).tolist()
        else:
            confs = results[0][4:, index].flatten()

        conf, label = np.max(confs), np.argmax(confs)
        return coords, conf, label

def predict(imgs_path, bb_conf = 0.5):
    model_path = 'final-resources/models/yolov8/best_full_integer_quant.tflite'
    precision = 'INT8'
    results_path = 'results/test'
    

    model = TFLiteModel(model_path, precision)
    final_detections = {}  # Dictionary to hold all detections

    for filename in os.listdir(imgs_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(imgs_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {filename}")
                continue

            # Process image for model prediction
            img_batch = preprocess_image(img, model.width, model.height, precision, convert_rgb=True)
            
            # Predict
            results = model.predict(img_batch)

            # Obtain all bounding boxes predictons in the image
            bb_dict = extract_bounding_boxes(bb_conf, model, img, results)

            # Apply NMS to reduce overlapping bounding boxes
            nms_filtered_dict = apply_nms(bb_dict)

            # Store results in the all_detections dictionary
            final_detections[filename] = nms_filtered_dict

            # # Draw the final bounding boxes on the image
            # final_img = draw_bounding_boxes_on_image(img, nms_filtered_dict)

            # # Save the final annotated image
            # save_path = os.path.join(results_path, filename)
            # cv2.imwrite(save_path, final_img)

    return final_detections

def txt_to_dict(filepath, image_width, image_height):
    """
    Read a .txt file containing object detections and convert it into a dictionary,
    converting normalized coordinates to pixel coordinates based on the image size.

    Args:
        filepath (str): The path to the .txt file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        dict: A dictionary with labels as keys and lists of coordinates in pixel values as values.
    """
    detections_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int((x_center - width / 2) * image_width)
                y1 = int((y_center - height / 2) * image_height)
                x2 = int((x_center + width / 2) * image_width)
                y2 = int((y_center + height / 2) * image_height)
                
                box = (x1, y1, x2, y2)
                
                if label in detections_dict:
                    detections_dict[label].append(box)
                else:
                    detections_dict[label] = [box]
            else:
                print("Warning: Line format incorrect ->", line)

    return detections_dict

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): The bounding box in format (x1, y1, x2, y2).
        box2 (tuple): The bounding box in format (x1, y1, x2, y2).

    Returns:
        float: The IoU value.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the union area by using both areas minus the intersection area
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou



def validate(detections, labels_path, imgs_path, results_path, results_filename='results'):
    """
    Validate predictions against ground truth data from .txt files.
    """
    

    data_df = []

    for img, _ in detections.items():
        print(img)
        base_name = os.path.splitext(img)[0]
        txt_file_name = base_name + '.txt'
        txt_file_path = os.path.join(labels_path, txt_file_name)

        image_path = os.path.join(imgs_path, img)
        if os.path.exists(image_path):
            # Obtain the dimensions of the image
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape

            if os.path.exists(txt_file_path):
                # Convert ground truth data to pixel coordinates
                groundTruth = txt_to_dict(txt_file_path, image_width, image_height)
                

                
                for gt_label in groundTruth:
                    for gt_coords in groundTruth[gt_label]:
                        # Recorro pred coords para comparar
                        try:
                            for pred_coords in detections[img][gt_label]:
                                iou_score = calculate_iou(gt_coords, pred_coords[:4])
                                data_df.append([img,
                                                gt_coords[0],  gt_coords[1], gt_coords[2], gt_coords[3],
                                                pred_coords[0], pred_coords[1], pred_coords[2], pred_coords[3], pred_coords[4],
                                                iou_score])
                        except KeyError as e:
                            print(f"Error: No se encontró la clave {e} en las detecciones.")
                            data_df.append([img,
                                            gt_coords[0],  gt_coords[1], gt_coords[2], gt_coords[3],
                                            None, None, None, None, None,
                                            None])


            else:
                print(f"No se encontró el archivo .txt para {img}")
        else:
            print(f"No se encontró la imagen {img}")


    # Crear el DataFrame con las columnas especificadas
    columns = ['image', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'pr_x1', 'pr_y1', 'pr_x2', 'pr_y2', 'pr_score', 'IoU']
    df = pd.DataFrame(data_df, columns=columns)
    df.to_csv(os.path.join(results_path, results_filename+ '.csv'), index=False)


if __name__ == "__main__":
    print('Predict')

    imgs_path = 'datasets/bioview-lizards_TRAIN/dataset/validation/images'
    detections = predict(imgs_path)



    print('Validate')
    
    validate(detections,
             labels_path = 'datasets/bioview-lizards_TRAIN/dataset/validation/labels',
             imgs_path = imgs_path,
             results_path='results/model-validation', results_filename='INT8')