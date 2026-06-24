# import re
# from paddleocr import TextRecognition

# ocr = TextRecognition(
#     model_name="th_PP-OCRv5_mobile_rec"
# )

# # ตัวที่ OCR ชอบสับสน -> เลข
# NUM_MAP = {
#     "O": "0",
#     "Q": "0",
#     "D": "0",
#     "I": "1",
#     "l": "1",
#     "|": "1",

#     "Z": "2",

#     "S": "5",

#     "B": "8",
# }

# # ไทยที่มักโดน OCR อ่านเป็นอังกฤษ
# THAI_MAP = {
#     "@": "ฮ",
#     "&": "ฃ",
#     "N": "ก",
#     "n": "ก"
# }

# PLATE_PATTERN = re.compile(
#     r'^[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}$'
# )


# def clean_plate(text):

#     # ลบช่องว่าง
#     text = text.strip()

#     # map อังกฤษ -> เลข
#     for old, new in NUM_MAP.items():
#         text = text.replace(old, new)

#     # map พิเศษ
#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)

#     # เก็บเฉพาะ ไทย อังกฤษ เลข
#     text = re.sub(r'[^A-Za-zก-๙0-9]', '', text)

#     # ลบอังกฤษที่เหลือ
#     text = re.sub(r'[A-Za-z]', '', text)

#     return text


# results = ocr.predict("img/2.jpg")

# for res in results:

#     if isinstance(res, dict):
#         text = res.get("rec_text", "")
#         score = res.get("rec_score", 0)
#     else:
#         text = str(res)
#         score = 0

#     plate = clean_plate(text)

#     print("-" * 50)
#     print(f"OCR Raw     : {text}")
#     print(f"Confidence  : {score}")
#     print(f"OCR Cleaned : {plate}")

#     if PLATE_PATTERN.match(plate):
#         print("VALID PLATE")
#     else:
#         print("INVALID PLATE")


# import re
# from paddleocr import PaddleOCR

# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="th_PP-OCRv5_mobile_rec",
# )

# NUM_MAP = {
#     "O": "0",
#     "Q": "0",
#     "D": "0",
#     "I": "1",
#     "l": "1",
#     "|": "1",
#     "Z": "2",
#     "S": "5",
#     "B": "8",
# }

# THAI_MAP = {
#     "@": "ฮ",
#     "&": "ฃ",
#     "N": "ก",
#     "n": "ก"
# }

# PLATE_PATTERN = re.compile(r'^[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}$')


# def clean_plate(text):
#     text = text.strip()

#     for old, new in NUM_MAP.items():
#         text = text.replace(old, new)

#     for old, new in THAI_MAP.items():
#         text = text.replace(old, new)

#     text = re.sub(r'[^A-Za-zก-๙0-9]', '', text)
#     text = re.sub(r'[A-Za-z]', '', text)

#     return text


# def run_ocr(img):
#     results = ocr.predict(img)

#     if not results:
#         return "", 0, "", False

#     res = results[0]

#     text = res.get("rec_text", "") if isinstance(res, dict) else str(res)
#     score = res.get("rec_score", 0) if isinstance(res, dict) else 0

#     plate = clean_plate(text)
#     valid = bool(PLATE_PATTERN.match(plate))

#     return text, score, plate, valid

import re
import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="th_PP-OCRv5_mobile_rec",
)

NUM_MAP = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1",
    "Z": "2", "S": "5", "B": "8"
}

THAI_MAP = {
    "@": "ฮ",
    "&": "ฃ",
    "N": "ก",
    "n": "ก"
}

PLATE_PATTERN = re.compile(r'^[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}$')


def preprocess(img):
    if img is None:
        return None

    h, w = img.shape[:2]

    if h < 80:
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    img = cv2.convertScaleAbs(img, alpha=1.5, beta=10)

    return img


def clean_plate(text):
    text = text.strip()

    for old, new in NUM_MAP.items():
        text = text.replace(old, new)

    for old, new in THAI_MAP.items():
        text = text.replace(old, new)

    text = re.sub(r'[^ก-๙0-9]', '', text)

    return text


def run_ocr(img):
    img = preprocess(img)

    if img is None or img.size == 0:
        return "", 0, "", False

    results = ocr.ocr(img)

    if not results or len(results[0]) == 0:
        return "", 0, "", False

    texts = []
    scores = []

    for line in results[0]:

        # 🔥 SAFE PARSE (กันทุก format)
        try:
            data = line[1]

            # case 1: [text, score]
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                txt = str(data[0])
                score = float(data[1])

            # case 2: string fallback
            else:
                txt = str(data)
                score = 0.0

        except Exception:
            continue

        texts.append(txt)
        scores.append(score)

    raw_text = "".join(texts)
    score = max(scores) if scores else 0

    plate = clean_plate(raw_text)
    valid = bool(PLATE_PATTERN.match(plate))

    return raw_text, score, plate, valid