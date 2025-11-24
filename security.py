import cv2
import pickle
import numpy as np
import time
import requests
import io  # <--- NEW IMPORT for in-memory image handling
from deepface import DeepFace

# ===========================
#   LOAD EMBEDDING
# ===========================
EMB_FILE = "my_embeddings.pkl"

try:
    with open(EMB_FILE, "rb") as f:
        avg_embedding = pickle.load(f)
    avg_embedding = np.array(avg_embedding)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    print("✔ Embedding loaded!")
except FileNotFoundError:
    print(f"❌ Error: '{EMB_FILE}' not found. Please run your training script first.")
    exit()

print("📸 Starting camera...")

# ===========================
#   TELEGRAM CONFIG
# ===========================
# ⚠️ SECURITY WARNING: Regenerate this token in BotFather after testing!
BOT_TOKEN = "bot_token"
CHAT_ID = "telegram_chat id"

def send_alert_with_photo(frame):
    """
    Encodes image to memory and sends to Telegram with a caption.
    No disk writing involved (prevents 'not downloadable' errors).
    """
    try:
        # 1. Encode frame to JPEG in memory (Buffer)
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if not ret:
            print("❌ Could not encode frame.")
            return

        # 2. Create an in-memory file object
        io_buf = io.BytesIO(buffer)
        
        # 3. Prepare the request
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        
        # The 'files' dict: (filename, file_object, mime_type)
        files = {
            'photo': ('alert.jpg', io_buf, 'image/jpeg')
        }
        
        # The 'data' dict: contains the chat_id and the text caption
        data = {
            'chat_id': CHAT_ID,
            'caption': "🚨 <b>ALERT! Unknown person was detected in your cabin!</b>", 
            'parse_mode': "HTML"
        }

        # 4. Send Request
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            print("✅ Alert & Photo sent successfully!")
        else:
            print(f"❌ Telegram Error: {response.text}")

    except Exception as e:
        print("❌ Connection Error:", e)

# ===========================
#   RECOGNITION SETTINGS
# ===========================
THRESHOLD = 0.5

UNKNOWN_TIME_LIMIT = 5   # seconds

unknown_start_time = None
alert_sent = False

cap = cv2.VideoCapture(0)

# ===========================
#   MAIN LOOP
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Copy frame for display/annotation so we don't send drawings to Telegram
    display_frame = frame.copy()

    try:
        # DeepFace detection
        emb = DeepFace.represent(
            img_path=frame,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False # Prevents crashing if no face found
        )

        # Check if a face was actually found
        if len(emb) > 0:
            emb_data = emb[0]["embedding"]
            emb_np = np.array(emb_data)
            emb_np = emb_np / np.linalg.norm(emb_np)

            cos_sim = np.dot(avg_embedding, emb_np)
            distance = 1 - cos_sim

            if distance < THRESHOLD:
                # --- Known person ---
                label = "ANUSH"
                color = (0, 255, 0)
                unknown_start_time = None
                alert_sent = False
            else:
                # --- Unknown person ---
                label = "UNKNOWN"
                color = (0, 0, 255)

                if unknown_start_time is None:
                    unknown_start_time = time.time()

                elapsed = time.time() - unknown_start_time
                countdown = UNKNOWN_TIME_LIMIT - int(elapsed)

                if countdown > 0:
                    label += f" | Alert in {countdown}s"

                if elapsed >= UNKNOWN_TIME_LIMIT and not alert_sent:
                    # Send the clean 'frame' (without text), not 'display_frame'
                    send_alert_with_photo(frame) 
                    alert_sent = True
        else:
            # No face detected by DeepFace
            label = "No Face"
            color = (255, 255, 0)
            unknown_start_time = None
            alert_sent = False

    except Exception as e:
        # DeepFace error handling
        label = "Scanning..."
        color = (255, 255, 0)
        unknown_start_time = None
        alert_sent = False
        # print(e) # Uncomment for debugging

    # Draw label on the display frame only
    cv2.putText(display_frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Intruder Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()