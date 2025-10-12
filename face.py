import cv2
import os
import json
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ====== CONFIG ======
SOURCE      = 0  # 0 = webcam หรือ URL/IP Camera
KNOWN_JSON  = "known_db.json"
TOLERANCE   = 0.5
FONT_SIZE   = 32
# ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, "font", "NotoSansThai_Condensed-Regular.ttf")

# โหลดฐานข้อมูลหลายมุมหลายภาพ
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

# วาดข้อความภาษาไทย
def put_text_thai(img, text, position, font_size=FONT_SIZE, color=(0,255,0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        print("⚠️ ไม่พบฟอนต์ ใช้ default font แทน")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==================== MAIN ====================
def main():
    known_encodings, known_info = load_known_faces()
    cap = cv2.VideoCapture(SOURCE)

    print("📷 Running... กด ESC เพื่อออก")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            name_text = "คนแปลกหน้า"

            if True in matches:
                match_idx = matches.index(True)
                person = known_info[match_idx]
                name_text = f"{person['name']} ({person['nickname']})"
                relation = person.get("relation", "")
                if relation:
                    name_text += f" - {relation}"

            # วาดกรอบรอบหน้า
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            frame = put_text_thai(frame, name_text, (left, top - 40))

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # กด ESC ออก
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

