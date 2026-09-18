<div align="center">

# 🤚 Hand Sign Recognition

**Real-time hand gesture detection and classification using Computer Vision**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-00A98F?logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Detect and classify 12+ hand gestures in real-time using your webcam*

---

</div>

## 📌 About

This project uses **Google's MediaPipe Hands** to detect 21 hand landmarks and classifies hand gestures based on finger positions. It works in real-time via webcam or on static images — no GPU required.

Built as a course project for **Computer Vision**.

## ✨ Features

- 🎥 **Real-time webcam detection** with live FPS counter
- 🖼️ **Static image mode** for single-image analysis
- 🤖 **12+ gesture recognition** using geometric finger-state analysis
- 🪞 **Mirror display** for natural interaction
- 📸 **Screenshot capture** with one keypress
- 🎨 **Clean HUD overlay** with gesture labels and hand info
- 🖐️ **Multi-hand support** — detects up to 2 hands simultaneously

## 🤝 Recognized Gestures

| Gesture | Emoji | Fingers Used |
|---------|:-----:|-------------|
| Fist | ✊ | All fingers closed |
| Open Hand | 🖐️ | All fingers extended |
| Peace / Victory | ✌️ | Index + Middle |
| Thumbs Up | 👍 | Only thumb (pointing up) |
| Thumbs Down | 👎 | Only thumb (pointing down) |
| Pointing | ☝️ | Only index finger |
| OK Sign | 👌 | Thumb + index tips touching, others open |
| Rock On | 🤘 | Index + Pinky |
| Call Me | 🤙 | Thumb + Pinky |
| Spider-Man | 🕷️ | Thumb + Index + Pinky |
| Three | 3️⃣ | Index + Middle + Ring |
| Four | 4️⃣ | All fingers except thumb |

## 🛠️ How It Works

```
Webcam Frame
    │
    ▼
┌─────────────────────┐
│  MediaPipe Hands    │ ──► 21 Hand Landmarks (x, y, z)
│  (Landmark Detection)│
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Finger State       │ ──► Which fingers are extended vs curled
│  Analysis           │     (thumb uses x-axis, others use y-axis)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Gesture            │ ──► Maps finger combinations to gesture names
│  Classification     │
└─────────────────────┘
    │
    ▼
  Annotated Frame + HUD
```

### Key Concepts

1. **Hand Landmark Detection** — MediaPipe detects 21 3D landmarks per hand in real-time
2. **Finger Extension Logic** — A finger is "extended" if its tip is above (lower y-value) its PIP joint; the thumb uses x-axis comparison instead
3. **Rule-Based Classification** — Gesture is determined by the combination of extended/curled fingers
4. **OK Sign Detection** — Uses Euclidean distance between thumb tip and index tip

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/chaitanya844/Hand-Sign-Recognition.git
cd Hand-Sign-Recognition

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Real-time webcam detection
python hand_sign_recognition.py

# Detect gestures in a static image
python hand_sign_recognition.py path/to/image.jpg
```

### Webcam Controls

| Key | Action |
|:---:|--------|
| `q` | Quit the application |
| `s` | Save a screenshot |

## 📁 Project Structure

```
Hand-Sign-Recognition/
│
├── hand_sign_recognition.py   # Main script — detection + classification
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── PROJECT REPORT.md          # Full project report
└── screenshots/               # Saved screenshots (created at runtime)
```

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image processing and webcam capture |
| `mediapipe` | Hand landmark detection (Google ML) |
| `numpy` | Numerical computations |

## ⚙️ Configuration

You can tune detection sensitivity by modifying these parameters in `hand_sign_recognition.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_num_hands` | 2 | Maximum hands to detect |
| `min_detection_confidence` | 0.6 | Minimum confidence for detection |
| `min_tracking_confidence` | 0.5 | Minimum confidence for tracking |

## 🧠 Adding New Gestures

To add a custom gesture, edit the `classify_gesture()` function:

```python
# Example: Detect a "Vulcan Salute" (🖖)
# Index+Middle together, Ring+Pinky together, thumb extended
if thumb and index and middle and ring and pinky:
    # Add additional logic for finger spacing
    return "Vulcan Salute", "🖖"
```

## 🔮 Future Improvements

- [ ] Add ASL (American Sign Language) alphabet recognition
- [ ] Implement gesture-based mouse control
- [ ] Train a custom ML model for more accurate classification
- [ ] Add gesture history / sequence detection
- [ ] Export detection data to CSV

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Google MediaPipe](https://mediapipe.dev/) — Hand landmark detection
- [OpenCV](https://opencv.org/) — Computer vision library

---

<div align="center">

**⭐ Star this repo if you found it useful!**

</div>
