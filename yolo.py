from ultralytics import YOLO
import cv2
import easyocr
import re
import warnings
from fastapi import UploadFile
import numpy as np


warnings.filterwarnings("ignore")
# Load model
model = YOLO("runs/detect/train/weights/best.pt")

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2"
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z"
}

def correct_pan(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    if len(text) < 10:
        return None

    text = text[:10]

    corrected = ""

    for i, char in enumerate(text):
        # First 5 characters must be letters
        if i < 5:
            if char.isdigit():
                char = DIGIT_TO_LETTER.get(char, char)

        # Next 4 must be digits
        elif 5 <= i <= 8:
            if char.isalpha():
                char = LETTER_TO_DIGIT.get(char, char)

        # Last must be letter
        else:
            if char.isdigit():
                char = DIGIT_TO_LETTER.get(char, char)

        corrected += char

    if re.fullmatch(r'[A-Z]{5}[0-9]{4}[A-Z]', corrected):
        return corrected

    return None

async def getPan(file:UploadFile) -> str: 
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.1)
    boxes = results[0].boxes.xyxy

    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        pan_crop = img[y1:y2, x1:x2]
    else:
        return "No PAN detected"
        
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

    # cv2.imshow("grey",thresh)
    # cv2.waitKey(0)

    result = reader.readtext(thresh)

    for r in result:
        text = r[1].replace(" ", "")
        final_pan = correct_pan(text)
        if final_pan:
            return final_pan
    else:
        return "PAN not extracted"
