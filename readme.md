# PAN Card Number Extraction System

## Overview

This project implements an end-to-end PAN card number extraction
pipeline using: - YOLOv8 for PAN region detection - OpenCV for image
preprocessing - EasyOCR for text extraction - Regex for PAN format
validation

The system detects the PAN number region from a PAN card image, extracts
the text, validates it, and returns the final structured output.

------------------------------------------------------------------------

## System Architecture

Input Image\
→ YOLO Detection\
→ Crop PAN Region\
→ OpenCV Preprocessing\
→ OCR (EasyOCR)\
→ Regex Validation\
→ Final PAN Output

------------------------------------------------------------------------

## Tools & Technologies

-   Python 3.x
-   Ultralytics YOLOv8
-   PyTorch (.pt model format)
-   Label Studio (annotation)
-   OpenCV
-   EasyOCR
-   Regular Expressions
-   Apple Silicon MPS (for training acceleration)

------------------------------------------------------------------------

## Dataset Structure

    pan-card/
    │
    ├── dataset/
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   │
    │   ├── labels/
    │   │   ├── train/
    │   │   └── val/
    │   │
    │   └── pan.yaml
    │
    ├── runs/
    │   └── detect/
    │       └── train/
    │           └── weights/
    │               ├── best.pt
    │               └── last.pt
    │
    └── test_images/

------------------------------------------------------------------------

## Annotation Configuration (Label Studio)

    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="pan_number" background="green"/>
      </RectangleLabels>
    </View>

Annotation Rules: - One bounding box per image - Tight bounding box
around the 10-character PAN number - Avoid extra background or nearby
fields

------------------------------------------------------------------------

## YOLO Configuration (pan.yaml)

    path: dataset
    train: images/train
    val: images/val

    names:
      0: pan_number

------------------------------------------------------------------------

## Model Training

Training Command:

    yolo task=detect mode=train \
         model=yolov8n.pt \
         data=pan.yaml \
         epochs=150 \
         imgsz=640 \
         batch=8 \
         device=mps \
         workers=0

Output model:

    runs/detect/train/weights/best.pt

------------------------------------------------------------------------

## Inference Pipeline

### 1. Load Model

``` python
from ultralytics import YOLO
model = YOLO("best.pt")
```

### 2. Detect PAN Region

``` python
results = model(img_path, conf=0.25)
boxes = results[0].boxes.xyxy
```

### 3. Crop Region

``` python
x1, y1, x2, y2 = map(int, boxes[0])
pan_crop = img[y1:y2, x1:x2]
```

------------------------------------------------------------------------

## Image Preprocessing (OpenCV)

Grayscale:

``` python
gray = cv2.cvtColor(pan_crop, cv2.COLOR_BGR2GRAY)
```

CLAHE Contrast Enhancement:

``` python
clahe = cv2.createCLAHE(2.0, (8,8))
contrast = clahe.apply(gray)
```

Adaptive Threshold:

``` python
thresh = cv2.adaptiveThreshold(...)
```

------------------------------------------------------------------------

## OCR Extraction

``` python
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(thresh)
```

------------------------------------------------------------------------

## PAN Validation

PAN Format:

    ABCDE1234F

Regex:

``` python
pattern = r'[A-Z]{5}[0-9]{4}[A-Z]'
```

Ensures correct format before returning output.

------------------------------------------------------------------------

## Performance Notes

-   10 images → Unstable confidence
-   150+ images → Stable detection
-   300+ images → Production-grade performance

Expected accuracy after scaling dataset: - Detection \> 90% - Final
extraction \> 95%

------------------------------------------------------------------------

## Future Improvements

-   Expand dataset to 500+ images
-   Add synthetic augmentation
-   Detect additional PAN card fields
-   Convert model to ONNX for deployment
-   Wrap into FastAPI service

------------------------------------------------------------------------

## Output Example

    PAN FOUND: ABCDE1234F

------------------------------------------------------------------------

## Conclusion

This project demonstrates: - Custom object detection training - Dataset
preparation and annotation - Model evaluation and debugging - OCR
integration - End-to-end document AI pipeline
