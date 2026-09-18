"""
Hand Sign Recognition - Computer Vision Project
=================================================
Real-time hand gesture recognition using MediaPipe Hands
and OpenCV. Detects and classifies common hand signs.

Recognized Gestures:
  - Fist, Open Hand, Peace (V), Thumbs Up,
    Thumbs Down, Pointing, OK Sign, Rock On,
    Three, Four

Usage:
  python hand_sign_recognition.py              -> webcam mode
  python hand_sign_recognition.py image.jpg    -> detect in an image
"""

import cv2
import mediapipe as mp
import sys
import os
import time
import math


# ─── MediaPipe Hands Setup ──────────────────────────────────────────────────

try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except (ImportError, AttributeError):
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError:
        from mediapipe.tasks.python import vision
        raise ImportError(
            "Incompatible MediaPipe version detected. Please run: pip install \"mediapipe<0.10.14\""
        )


# ─── Finger State Detection ────────────────────────────────────────────────

def get_finger_states(hand_landmarks, handedness):
    """
    Determine which fingers are extended (open) or curled (closed).

    Parameters:
        hand_landmarks: MediaPipe hand landmarks.
        handedness: 'Left' or 'Right' hand label.

    Returns:
        dict with finger names mapped to True (extended) / False (curled).
    """
    landmarks = hand_landmarks.landmark

    # Tip and PIP (proximal interphalangeal) joint indices
    # A finger is "extended" if its tip is above (lower y) its PIP joint
    finger_tips = {
        "index":  (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        "middle": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        "ring":   (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        "pinky":  (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    }

    fingers = {}

    # Thumb: compare x-axis (depends on which hand)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    if handedness == "Right":
        fingers["thumb"] = thumb_tip.x < thumb_ip.x
    else:
        fingers["thumb"] = thumb_tip.x > thumb_ip.x

    # Other fingers: compare y-axis (tip above PIP = extended)
    for name, (tip_id, pip_id) in finger_tips.items():
        fingers[name] = landmarks[tip_id].y < landmarks[pip_id].y

    return fingers


def thumb_is_up(hand_landmarks):
    """Check if the thumb is pointing upward (tip well above MCP)."""
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.y < thumb_mcp.y and thumb_tip.y < index_mcp.y


def thumb_is_down(hand_landmarks):
    """Check if the thumb is pointing downward."""
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return thumb_tip.y > thumb_mcp.y and thumb_tip.y > wrist.y


def ok_sign(hand_landmarks):
    """Detect OK sign: thumb tip close to index tip, other fingers extended."""
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )
    return distance < 0.05


# ─── Gesture Classification ────────────────────────────────────────────────

def classify_gesture(hand_landmarks, handedness):
    """
    Classify the hand gesture based on finger states.

    Parameters:
        hand_landmarks: MediaPipe hand landmarks.
        handedness: 'Left' or 'Right'.

    Returns:
        gesture_name (str): Name of the detected gesture.
        emoji (str): Emoji representing the gesture.
    """
    fingers = get_finger_states(hand_landmarks, handedness)

    thumb = fingers["thumb"]
    index = fingers["index"]
    middle = fingers["middle"]
    ring = fingers["ring"]
    pinky = fingers["pinky"]

    extended_count = sum([thumb, index, middle, ring, pinky])

    # Check special gestures first

    # OK Sign: thumb and index tips touching, others extended
    if ok_sign(hand_landmarks) and middle and ring and pinky:
        return "OK Sign", "👌"

    # Thumbs Up: only thumb extended, pointing up
    if thumb and not index and not middle and not ring and not pinky:
        if thumb_is_up(hand_landmarks):
            return "Thumbs Up", "👍"
        elif thumb_is_down(hand_landmarks):
            return "Thumbs Down", "👎"

    # Fist: no fingers extended
    if extended_count == 0:
        return "Fist", "✊"

    # Pointing: only index extended
    if not thumb and index and not middle and not ring and not pinky:
        return "Pointing", "☝️"

    # Peace / Victory: index + middle extended
    if not thumb and index and middle and not ring and not pinky:
        return "Peace", "✌️"

    # Rock On: index + pinky extended
    if not thumb and index and not middle and not ring and pinky:
        return "Rock On", "🤘"

    # Three: index + middle + ring
    if not thumb and index and middle and ring and not pinky:
        return "Three", "3️⃣"

    # Four: all except thumb
    if not thumb and index and middle and ring and pinky:
        return "Four", "4️⃣"

    # Open Hand / Wave: all fingers extended
    if extended_count == 5:
        return "Open Hand", "🖐️"

    # Spider-Man: thumb + index + pinky
    if thumb and index and not middle and not ring and pinky:
        return "Spider-Man", "🕷️"

    # Call Me: thumb + pinky
    if thumb and not index and not middle and not ring and pinky:
        return "Call Me", "🤙"

    return "Unknown", "❓"


# ─── Drawing Helpers ────────────────────────────────────────────────────────

# Color palette
COLORS = {
    "bg":       (30, 30, 30),
    "green":    (0, 230, 118),
    "blue":     (255, 160, 50),
    "yellow":   (0, 230, 255),
    "white":    (255, 255, 255),
    "gray":     (150, 150, 150),
    "dark":     (50, 50, 50),
}


def draw_hud(frame, gesture_name, emoji, hand_label, fps, hand_count):
    """Draw a clean heads-up display on the frame."""
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), COLORS["bg"], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Gesture name (large)
    cv2.putText(
        frame, f"{emoji}  {gesture_name}",
        (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["green"], 2,
    )

    # FPS and hand info (right side)
    cv2.putText(
        frame, f"FPS: {fps:.0f}",
        (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["yellow"], 1,
    )
    cv2.putText(
        frame, f"Hands: {hand_count} ({hand_label})",
        (w - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["gray"], 1,
    )

    # Bottom instruction bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), COLORS["bg"], -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(
        frame, "Q: Quit | S: Screenshot | Show hand signs to the camera!",
        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["gray"], 1,
    )

    return frame


# ─── Process a Single Frame ────────────────────────────────────────────────

def process_frame(frame, hands):
    """
    Detect hands and classify gestures in a single frame.

    Parameters:
        frame (numpy.ndarray): BGR image.
        hands: MediaPipe Hands object.

    Returns:
        annotated frame, gesture name, hand count
    """
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_name = "No Hand"
    emoji = "🤷"
    hand_label = "-"
    hand_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_count = len(results.multi_hand_landmarks)

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Get handedness (MediaPipe mirrors labels for front camera)
            hand_label = hand_info.classification[0].label
            confidence = hand_info.classification[0].score

            # Classify gesture
            gesture_name, emoji = classify_gesture(hand_lm, hand_label)

            # Draw bounding box around hand
            lm_list = hand_lm.landmark
            x_coords = [lm.x for lm in lm_list]
            y_coords = [lm.y for lm in lm_list]
            h, w = frame.shape[:2]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLORS["green"], 2)

            # Label on bounding box
            cv2.putText(
                frame, f"{hand_label} - {confidence:.0%}",
                (x_min, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["blue"], 1,
            )

    return frame, gesture_name, emoji, hand_label, hand_count


# ─── Mode 1: Static Image Detection ────────────────────────────────────────

def detect_in_image(image_path):
    """Detect hand signs in a static image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Processing: {image_path}")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        result, gesture, emoji, hand_label, count = process_frame(img, hands)
        print(f"[INFO] Hands: {count} | Gesture: {gesture}")

        out_path = "output_" + os.path.basename(image_path)
        result = draw_hud(result, gesture, emoji, hand_label, 0, count)
        cv2.imwrite(out_path, result)
        print(f"[INFO] Saved: {out_path}")

        cv2.imshow("Hand Sign Detection - Press any key", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─── Mode 2: Real-Time Webcam Detection ────────────────────────────────────

def detect_from_webcam():
    """Real-time hand sign detection from webcam."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("[INFO] Webcam opened. Show hand signs to the camera!")
    print("[INFO] Press 'q' to quit, 's' for screenshot.")

    prev_time = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            # Process
            result, gesture, emoji, hand_label, count = process_frame(frame, hands)

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            # Draw HUD
            result = draw_hud(result, gesture, emoji, hand_label, fps, count)

            cv2.imshow("Hand Sign Recognition", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"handsign_{int(time.time())}.jpg"
                cv2.imwrite(fname, result)
                print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_in_image(sys.argv[1])
    else:
        detect_from_webcam()
