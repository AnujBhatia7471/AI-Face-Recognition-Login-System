**Face Recognition Login Page**
A browser-based face recognition authentication system built using Python, Flask, OpenCV, and ArcFace (ONNX).
The application allows users to register and log in using their face via a webcam, removing reliance on traditional password-only authentication.
This project demonstrates a real-world ML deployment workflow, combining computer vision, deep learning inference, backend APIs, and frontend camera integration.

**🚀 Key Features**
Browser-based face registration and login
No terminal or CLI interaction for users
High-accuracy ArcFace ONNX model for facial embeddings
Cosine similarity–based face matching
OpenCV DNN–based face detection
Secure storage of facial embeddings (not images)
SQLite database for persistence
Flask REST API backend
Automatic ONNX model download at runtime
Clean and responsive frontend UI
Separation of frontend and backend (deployment-ready)

**🛠 Technology Stack**
**Backend**
Python 3.10
Flask
Flask-CORS
OpenCV (headless)
ONNX Runtime
ArcFace (ONNX)
NumPy
SQLite
**Frontend**
HTML
CSS
JavaScript (Webcam + Fetch API)

**📁 Project Structure**
<img width="416" height="511" alt="image" src="https://github.com/user-attachments/assets/f0200948-62ff-4b92-865e-dc1c98171897" />



**Note:**
The ArcFace ONNX model is not stored in GitHub. It is automatically downloaded at runtime.

**⚙️ Installation (Local Setup)**
1️⃣ Clone the repository
git clone https://github.com/AnujBhatia7471/Face-Recognition-Login-Page.git
cd Face-Recognition-Login/Backend
2️⃣ Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the backend
python app.py

Backend runs at:
http://127.0.0.1:5000
5️⃣ Run the frontend
cd ../Frontend
python -m http.server 5500


Open in browser:
http://127.0.0.1:5500

**🔄 Application Workflow**
🔹 Face Registration
User enters an email
Webcam is activated in the browser
Multiple face samples are captured
ArcFace generates facial embeddings
Embeddings are stored securely in the database

**🔹 Face Login**
User enters registered email
Webcam captures a live face image
ArcFace generates a new embedding
Cosine similarity is computed
Login is approved only if similarity exceeds a safe threshold

**🔐 Security Considerations**
Facial embeddings are stored — not raw images
No face images are saved on disk
Strict similarity threshold to prevent false matches
Single-face detection enforced per frame
Designed for educational and demonstration purposes

**📌 Notes & Limitations**
Camera permission must be enabled in the browser
Best performance under good lighting conditions
Webcam access requires:
http://localhost (local)
HTTPS (production deployment)
Multiple faces in a single frame are not supported
Liveness detection (anti-spoofing) is not included

**🌍 Deployment**
This project is deployment-ready using a split architecture:
Backend (ML + API): Railway / Fly.io / Cloud VM
Frontend (static): Netlify / Vercel
⚠️ HTTPS is mandatory in production for browser camera access.

**📄 License**
This project is intended for educational and learning purposes only.
Not recommended for production authentication without additional security layers.

**👤 Author**
Anuj Bhatia
GitHub: https://github.com/AnujBhatia7471

**🧠 Summary**
This project demonstrates a complete facial authentication pipeline, integrating:
Computer vision
Deep learning inference
REST APIs
Secure data handling
Frontend webcam integration
Real deployment architecture
It is suitable for:
Academic projects
ML / CV portfolios
Backend + ML interview discussions
