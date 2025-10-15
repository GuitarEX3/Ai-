import cv2
import os
import json
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ====== CONFIG ======
SOURCE = 0
KNOWN_JSON = "known_db.json"
TOLERANCE = 0.5
FONT_SIZE = 30
DNN_CONFIDENCE = 0.4
# ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, "font", "NotoSansThai_Condensed-Regular.ttf")

MODEL = os.path.join(SCRIPT_DIR, "deploy.prototxt")
WEIGHTS = os.path.join(SCRIPT_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(MODEL, WEIGHTS)

# ==================== โหลดฐานข้อมูล ====================
def load_known_faces():
    if not os.path.exists(KNOWN_JSON):
        print("⚠️ ไม่พบไฟล์ฐานข้อมูล:", KNOWN_JSON)
        return [], []

    with open(KNOWN_JSON, "r", encoding="utf-8") as f:
        known_data = json.load(f)

    known_encodings, known_info = [], []
    for person in known_data:
        for img_path in person.get("images", []):
            img_full = os.path.join(SCRIPT_DIR, img_path)
            if os.path.exists(img_full):
                img = face_recognition.load_image_file(img_full)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_info.append({
                        "name": person.get("name", ""),
                        "nickname": person.get("nickname", ""),
                        "relation": person.get("relation", "")
                    })
    print(f"✅ โหลดฐานข้อมูล {len(known_encodings)} encoding สำเร็จ")
    return known_encodings, known_info

# ==================== วาดกรอบชื่อใต้ใบหน้า ====================
def draw_name_box(frame, left, top, right, bottom, lines, color=(0, 255, 0)):
    overlay = frame.copy()
    line_height = FONT_SIZE + 6
    box_height = line_height * len(lines) + 10
    box_y1 = bottom
    box_y2 = bottom + box_height

    # กล่องโปร่งใสใต้หน้า
    cv2.rectangle(overlay, (left, box_y1), (right, box_y2), color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # กรอบใบหน้า
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (left-2, top-2), (right+2, bottom+2), (0,0,0), 1)

    # เขียนข้อความหลายบรรทัด
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    text_color = (255, 255, 255)
    y = bottom + 5
    for line in lines:
        draw.text((left + 10, y), line, font=font, fill=text_color)
        y += line_height
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==================== ปรับแสง/Contrast ====================
def enhance_frame_color(frame, alpha=1.15, beta=10):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# ==================== MAIN ====================
def main():
    known_encodings, known_info = load_known_faces()
    cap = cv2.VideoCapture(SOURCE)
    print("📷 Running... กด ESC เพื่อออก")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = enhance_frame_color(frame)
        (h, w) = frame.shape[:2]

        # ตรวจจับใบหน้า
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_locations = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DNN_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                face_locations.append((y1, x2, y2, x1))

        if len(face_locations) == 0:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)

        rgb_frame = frame[:, :, ::-1]
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)

            if True in matches:
                match_idx = matches.index(True)
                person = known_info[match_idx]
                lines = [person['name'], f"({person['nickname']})"]
                if person.get("relation"):
                    lines.append(f"- {person['relation']}")
                color = (0, 255, 0)
            else:
                lines = ["คนแปลกหน้า"]
                color = (0, 0, 255)

            frame = draw_name_box(frame, left, top, right, bottom, lines, color=color)

        cv2.imshow("AI Vision Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

