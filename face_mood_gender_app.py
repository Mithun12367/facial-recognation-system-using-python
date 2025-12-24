import cv2
import face_recognition
import pickle
from fer import FER

# ================= LOAD KNOWN FACES =================
print("Loading embeddings...")
data = pickle.load(open("embeddings.pkl", "rb"))
known_encodings = [d["encoding"] for d in data]
known_names = [d["name"] for d in data]

# ================= EMOTION DETECTOR =================
emotion_detector = FER(mtcnn=False)

# ================= GENDER MODEL =================
gender_prototxt = "models/gender/deploy_gender.prototxt"
gender_model = "models/gender/gender_net.caffemodel"
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)
gender_list = ["Male", "Female"]

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit or close the window.")

# ================= COLORS =================
mood_colors = {
    "angry": (0, 0, 255),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprise": (255, 255, 0),
    "neutral": (180, 180, 180)
}

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera frame.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):

        # ---------- FACE RECOGNITION ----------
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=0.5
        )
        name = "UNKNOWN"
        if True in matches:
            name = known_names[matches.index(True)]

        # ---------- SAFE FACE CROP ----------
        h, w = frame.shape[:2]
        pad = 10
        t = max(0, top - pad)
        l = max(0, left - pad)
        b = min(h, bottom + pad)
        r = min(w, right + pad)
        face_crop = frame[t:b, l:r].copy()

        # ---------- EMOTION ----------
        mood = "neutral"
        try:
            emotions = emotion_detector.detect_emotions(face_crop)
            if emotions:
                mood = max(
                    emotions[0]["emotions"],
                    key=emotions[0]["emotions"].get
                )
        except Exception:
            mood = "neutral"

        color = mood_colors.get(mood, (255, 255, 255))

        # ---------- GENDER ----------
        try:
            blob = cv2.dnn.blobFromImage(
                face_crop,
                1.0,
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False,
                crop=False
            )
            gender_net.setInput(blob)
            gender = gender_list[gender_net.forward()[0].argmax()]
        except Exception:
            gender = "Unknown"

        # ---------- DRAW BOX ----------
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = f"{name} | {gender} | {mood}"
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.rectangle(
            frame,
            (left, top - lh - 10),
            (left + lw + 10, top),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (left + 5, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    # ---------- SHOW ----------
    cv2.imshow("Face + Mood + Gender", frame)

    # Exit on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Q pressed, exiting...")
        break

    # Exit if window closed manually
    if cv2.getWindowProperty("Face + Mood + Gender", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed, exiting...")
        break

# ================= CLEAN EXIT =================
cap.release()
cv2.destroyAllWindows()
print("✅ Camera released and program closed properly")
