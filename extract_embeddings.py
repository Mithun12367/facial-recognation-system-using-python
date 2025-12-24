# extract_embeddings.py
import os
import pickle
import face_recognition

DATASET_DIR = "dataset"
OUTPUT_FILE = "embeddings.pkl"

data = []
print("🔍 Extracting face embeddings from dataset...")

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        print("Processing:", img_path)

        image = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(image)

        if len(encs) > 0:
            data.append({"name": person_name, "encoding": encs[0]})
            print("  ✔ saved encoding for", person_name)
        else:
            print("  ⚠ no face found in", img_path)

pickle.dump(data, open(OUTPUT_FILE, "wb"))
print("✅ Embeddings saved to", OUTPUT_FILE)
