import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

DATASET_DIR = "dataset"
OUTPUT_PKL = "embeddings.pkl"

all_embeddings = {}

print("📌 Loading DeepFace ArcFace model for embeddings...\n")

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"➡ Processing: {person_name}")

    person_embeds = []

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        try:
            embedding_info = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                detector_backend="opencv"     # faster detection
            )[0]["embedding"]

            person_embeds.append(np.array(embedding_info))

        except Exception as e:
            print(f"   ❌ Face not detected in: {img_name}")
            continue

    if len(person_embeds) == 0:
        print(f"   ⚠ No valid faces found for {person_name}, skipping.\n")
        continue

    # Average embedding for this person
    avg_emb = np.mean(person_embeds, axis=0)
    all_embeddings[person_name] = avg_emb

    print(f"   ✔ Saved {len(person_embeds)} face embeddings for {person_name}\n")

# Save to PKL
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(all_embeddings, f)

print("🎉 DONE! Embeddings saved to:", OUTPUT_PKL)
