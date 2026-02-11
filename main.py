from ultralytics import YOLO
import cv2
import easyocr
import re
import warnings

warnings.filterwarnings("ignore")
# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Read image
img_path = "test_images/pan.jpg"
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

thresh = cv2.adaptiveThreshold(
    contrast,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    2
)

reader = easyocr.Reader(['en'], gpu=False)

result = reader.readtext(thresh)

pattern = r'[A-Z]{5}[0-9]{4}[A-Z]'

for r in result:
    text = r[1].replace(" ", "")
    
    if re.fullmatch(pattern, text):
        print("PAN FOUND:", text)
        break
else:
    print("PAN not extracted")
