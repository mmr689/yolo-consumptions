import cv2
# import matplotlib.pyplot as plt

from ultralytics import YOLO

model = YOLO('datasets/bioview-lizards_TRAIN/run/train/weights/best.pt')

img_path = 'datasets/bioview-lizards_TRAIN/dataset/validation/images/img_20230729_223004.jpg'
img = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model.predict(img_path)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score >= 0.5:
        cv2.rectangle(img,
                      (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 3)
        cv2.putText(img,
                    str(round(score,1)),
                    (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

cv2.imwrite('result.jpg', img)