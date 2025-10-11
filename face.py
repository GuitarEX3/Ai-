import cv2
import os
import json
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ====== CONFIG ======
SOURCE      = 0                                   # 0 = webcam หรือ URL/IP Camera
KNOWN_JSON  = "known_db.json"                     # ฐานข้อมูลคนรู้จัก
TOLERANCE   = 0.5                                 # ค่าความคล้ายใบหน้า
FONT_PATH   = "font/NotoSansThai_Condensed-Regular.ttf"  # ฟอนต์ภาษาไทย (ต้องมีไฟล์)
FONT_SIZE   = 32
# ====================

# โหลดฐานข้อมูลคนรู้จัก
def load_known_faces():
    if not os.path.exists(KNOWN_JSON):
        print("⚠️ ไม่พบไฟล์ฐานข้อมูล:", KNOWN_JSON)
        return [], []

    with open(KNOWN_JSON, "r", encoding="utf-8") as f:
        known_data = json.load(f)

    known_encodings, known_info = [], []
    for person in known_data:
        if os.path.exists(person["image"]):
            img = face_recognition.load_image_file(person["image"])
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_info.append({
                    "name": person["name"],
                    "nickname": person.get("nickname", ""),
                    "relation": person.get("relation", "")
                })

    print(f"✅ โหลดฐานข้อมูล {len(known_info)} คนสำเร็จ")
    return known_encodings, known_info


# วาดข้อความภาษาไทยบน OpenCV image
def put_text_thai(img, text, position, font_path=FONT_PATH, font_size=FONT_SIZE, color=(0,255,0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
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

        rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
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

            # แสดงชื่อ + ความสัมพันธ์
            frame = put_text_thai(frame, name_text, (left, top - 40))

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # กด ESC ออก
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
