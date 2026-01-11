const API_BASE = "http://localhost:5000";
document.getElementById("regForm").addEventListener("submit", e => {
  e.preventDefault();
});


const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const sampleBtn = document.getElementById("sampleBtn");
const startBtn = document.getElementById("startBtn");
startBtn.onclick = startRegistration;
sampleBtn.onclick = takeSample;

window.onbeforeunload = null;

let sampleCount = 0;
const MAX_SAMPLES = 5;

window.onbeforeunload = function() {
  return null;
};


async function startRegistration() {
  if (!email.value || !password.value) {
    result.innerText = "Email and password required";
    return;
  }

  await startCamera();
  sampleBtn.disabled = false;
  startBtn.disabled = true;
  result.innerText = "Camera started";
}

function takeSample() {
  if (sampleCount >= MAX_SAMPLES) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const f = new FormData();
    f.append("email", email.value.trim());
    f.append("password", password.value);
    f.append("image", blob);

    fetch(`${API_BASE}/register`, { method: "POST", body: f })
      .then(r => r.json())
      .then(d => {
        // Always show message
        result.innerText = d.msg;
        result.style.color = d.success ? "green" : "red";
        result.style.fontWeight = "600";

        // ❌ If face not detected or error → STOP here
        if (!d.success) {
          // Keep camera ON
          // Do NOT increase sample count
          // Do NOT reset UI
          return;
        }

        // ✅ Face was saved
        sampleCount++;

        if (d.completed) {
          stopCamera();
          result.innerText = "Registration completed";
          sampleBtn.disabled = true;
        } else {
          sampleBtn.innerText = `Take Sample (${sampleCount + 1} / 5)`;
        }
      })
      .catch(() => {
        result.innerText = "Server error";
        result.style.color = "red";
      });
  }, "image/jpeg", 0.95);
}
