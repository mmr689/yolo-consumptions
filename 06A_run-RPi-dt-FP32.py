import cv2

from ultralytics import YOLO

model = YOLO('run/best.pt')

img_path = 'run/test/images/img_20230101_191503.jpg'
img = cv2.imread(img_path)

results = model.predict(img_path)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score >= 0:
        cv2.rectangle(img,
                      (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 3)
        cv2.putText(img,
                    str(round(score,1)),
                    (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

cv2.imwrite('result.jpg', img)