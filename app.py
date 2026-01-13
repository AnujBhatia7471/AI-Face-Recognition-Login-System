import os
import sqlite3
import numpy as np
import urllib.request
import threading
import os
import time
cv2 = None
ort = None
face_detector = None



from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")


from flask_cors import CORS
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/login")
def login_page():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")




def get_cv2():
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    return cv2




def get_face_detector():
    global cv2, face_detector

    if face_detector is None:
        import cv2 as _cv2
        cv2 = _cv2

        if not os.path.exists(CAFFE_PATH):
            urllib.request.urlretrieve(CAFFE_URL, CAFFE_PATH)

        face_detector = cv2.dnn.readNetFromCaffe(
            os.path.join(BASE_DIR, "deploy.prototxt"),
            CAFFE_PATH
        )

    return face_detector






def detect_face(img):
    if img is None:
        return None

    cv = get_cv2()
    detector = get_face_detector()

    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(
        img, 1.0, (300, 300), (104, 177, 123)
    )

    detector.setInput(blob)
    dets = detector.forward()

    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] > 0.9:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = img[y1:y2, x1:x2]
            if face.size:
                return face
    return None





# ---------- CORS (LOCKED) ----------
CORS(app)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= FACE DETECTOR =================
CAFFE_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

CAFFE_PATH = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
face_detector = None



# ================= DATABASE =================
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn, conn.cursor()

conn, cur = get_db()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()
conn.close()

# ================= ARC FACE =================
# MODEL_URL = (
#     "https://huggingface.co/FoivosPar/Arc2Face/resolve/"
#     "da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx"
# )
MODEL_PATH = os.path.join(BASE_DIR, "arcface_fp16.onnx")

arcface = None
arc_input_name = None
arc_lock = threading.Lock()

def get_arcface():
    global arcface, arc_input_name
    with arc_lock:
        if arcface is None:
            arcface = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
            arc_input_name = arcface.get_inputs()[0].name
    return arcface

# ================= UTILS =================
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_arcface():
    global ort, arcface, arc_input_name

    if arcface is None:
        import onnxruntime as _ort
        ort = _ort

        arcface = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        arc_input_name = arcface.get_inputs()[0].name

    return arcface


def get_embedding(face):
    cv = get_cv2()
    face = cv.resize(face, (112, 112))
    face = cv.cvtColor(face, cv.COLOR_BGR2RGB)


    face = (face.astype(np.float32) / 255.0).astype(np.float16)
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    session = get_arcface()
    outs = session.run(None, {arc_input_name: face})

    # Pick correct ArcFace embedding (prevents 256 vs 512 bug)
    for o in outs:
        if len(o.shape) == 2 and o.shape[1] in (256, 512):
            emb = o[0]
            break
    else:
        raise Exception("ArcFace embedding not found")

    emb = emb.astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb




# ================= CONFIG =================
THRESHOLD = float(os.environ.get("FACE_THRESHOLD", 0.32))

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():
    try:
        email = request.form.get("email", "").lower()
        password = request.form.get("password", "")
        image = request.files.get("image")

        if not email or not password or not image:
            return jsonify(success=False, msg="Missing data")

        conn, cur = get_db()

        cur.execute("SELECT COUNT(*) FROM embeddings WHERE email=?", (email,))
        if cur.fetchone()[0] >= 5:
            conn.close()
            return jsonify(success=False, msg="Already fully registered")

        cur.execute("SELECT email FROM users WHERE email=?", (email,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, password)
            )

        cv = get_cv2()
        img = cv.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv.IMREAD_COLOR
        )


        face = detect_face(img)
        if face is None:
            conn.close()
            return jsonify(success=False, msg="No face detected")

        emb = get_embedding(face)

        cur.execute(
            "INSERT INTO embeddings (email, embedding) VALUES (?, ?)",
            (email, emb.tobytes())
        )

        cur.execute("SELECT COUNT(*) FROM embeddings WHERE email=?", (email,))
        count = cur.fetchone()[0]

        conn.commit()
        conn.close()

        return jsonify(
            success=True,
            completed=(count == 5),
            msg="Registration completed" if count == 5 else f"Face saved ({count}/5)"
        )

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify(success=False, msg="Internal server error")

# ================= LOGIN (FACE) =================
@app.route("/login/face", methods=["POST"])
def face_login():
    try:
        email = request.form.get("email")
        image = request.files.get("image")

        if not email or not image:
            return jsonify(success=False, msg="Missing data")

        conn, cur = get_db()
        cur.execute("SELECT embedding FROM embeddings WHERE email=?", (email,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return jsonify(success=False, msg="User not registered")

        cv = get_cv2()
        img = cv.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv.IMREAD_COLOR
        )


        face = detect_face(img)
        if face is None:
            return jsonify(success=False, msg="No face detected")

        emb = get_embedding(face)

        scores = [
            cosine_sim(emb, np.frombuffer(r[0], dtype=np.float32))
            for r in rows
        ]

        best_score = max(scores)

        if best_score >= THRESHOLD:
            return jsonify(success=True, msg="Login successful")

        return jsonify(success=False, msg="Face does not match")

    except Exception as e:
        print("FACE LOGIN ERROR:", e)
        return jsonify(success=False, msg="Internal server error")

# ================= LOGIN (PASSWORD) =================
@app.route("/login/password", methods=["POST"])
def password_login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify(success=False, msg="Missing credentials")

        conn, cur = get_db()
        cur.execute("SELECT password FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify(success=False, msg="User not found")

        if row[0] != password:
            return jsonify(success=False, msg="Invalid password")

        return jsonify(success=True, msg="Login successful")

    except Exception as e:
        print("PASSWORD LOGIN ERROR:", e)
        return jsonify(success=False, msg="Internal server error")

# ================= MAIN =================
if __name__ == "__main__":
    import threading

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
