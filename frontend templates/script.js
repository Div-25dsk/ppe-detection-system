document.getElementById("confidence").addEventListener("input", function () {
  document.getElementById("confidenceValue").textContent = this.value;
});

document.getElementById("startBtn").addEventListener("click", function () {
  // Replace with your backend streaming URL
  document.getElementById("videoStream").src = "http://127.0.0.1:5000/video_feed";
});

document.getElementById("stopBtn").addEventListener("click", function () {
  document.getElementById("videoStream").src = "";
});

document.getElementById("setClassesBtn").addEventListener("click", function () {
  const selected = Array.from(document.getElementById("classes").selectedOptions).map(opt => opt.value);
  alert("Classes set to detect: " + selected.join(", "));
});
