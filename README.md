**AI Face Recognition Login System**

An AI-powered authentication system that allows users to log in using face recognition instead of traditional passwords. This project integrates computer vision and machine learning into a real-time web application using Python and Flask.

Features

Real-time face detection using OpenCV

Face recognition using ArcFace ONNX model

Password-less authentication

Secure storage of facial embeddings

Multi-sample registration for improved accuracy

Optional password-based login fallback

Web interface built with Flask and JavaScript

Tech Stack

Backend: Python, Flask

Frontend: HTML, CSS, JavaScript

Computer Vision: OpenCV

Machine Learning: ArcFace ONNX model, ONNX Runtime

Database: SQLite

Other Tools: NumPy, psutil

How It Works

User Registration

User provides email, password, and face images.

System detects the face using OpenCV.

ArcFace model generates a facial embedding.

Embeddings are stored in the database.

Face Login

User submits a face image.

System extracts the embedding.

Cosine similarity compares it with stored embeddings.

If similarity exceeds the threshold, login is successful.

<img width="248" height="372" alt="image" src="https://github.com/user-attachments/assets/ee91c466-e927-422b-b016-f7adcf2d794a" />

Installation
1. Clone the repository
git clone https://github.com/yourusername/face-recognition-login.git
cd face-recognition-login

2. Create virtual environment (recommended)
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


If requirements file is not present, install manually:

pip install flask opencv-python numpy onnxruntime psutil flask-cors

4. Run the application
python app.py


Open in browser:

http://localhost:5000

Configuration

You can adjust the face recognition threshold:

FACE_THRESHOLD = 0.32

Security Notes

Passwords are stored in plaintext (for demo purposes).

In production:

Use password hashing (bcrypt).

Use HTTPS.

Add liveness detection.

Future Improvements

Liveness detection to prevent spoofing

Cloud deployment (AWS/GCP)

Role-based authentication

Face recognition optimization with GPU

Author

Anuj Bhatia
Computer Science Engineer
Portfolio: https://anujbhatia.pythonanywhere.com/
