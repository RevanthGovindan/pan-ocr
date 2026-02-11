from ultralytics import YOLO
import cv2
import easyocr
import re
import warnings

warnings.filterwarnings("ignore")
# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Read image
img_path = "test_images/ranjith.jpeg"
img = cv2.imread(img_path)

# Run detection
results = model(img_path, conf=0.25)

boxes = results[0].boxes.xyxy

if len(boxes) > 0:
    x1, y1, x2, y2 = map(int, boxes[0])
    pan_crop = img[y1:y2, x1:x2]
else:
    print("No PAN detected")
    exit(0)
    
gray = cv2.cvtColor(pan_crop, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(gray)

blur = cv2.GaussianBlur(contrast, (3, 3), 0)

# Otsu Thresholding
_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

reader = easyocr.Reader(['en'], gpu=True)

cv2.imshow("grey",thresh)
cv2.waitKey(0)

result = reader.readtext(thresh)

pattern = r'[A-Z]{5}[0-9]{4}[A-Z]'

for r in result:
    text = r[1].replace(" ", "")
    
    if re.fullmatch(pattern, text):
        print("PAN FOUND:", text)
        break
else:
    print("PAN not extracted")
