import cv2
import mediapipe as mp
import time
import math
import os

# ---- Config ----
IMAGE_FOLDER = "Images"
map_images = {
    "both_hands_open_mouth": "BothhandsWithopenmouth.png",
    "no_hand": "NoHandGesture.png",
    "point_to_mouth": "PointFingerMouth.png",
    "point_up_open_mouth": "PointFingerWithMoutOpen.png",
}
HOLD_TIME = 0.4  # seconds gesture must be held before showing image
MOUTH_PROX_THRESHOLD = 0.12  # for point to mouth detection
BOTH_HANDS_DIST_THRESHOLD = 0.45  # increased even more for easier detection
MOUTH_OPEN_RATIO = 0.12  # made more sensitive for mouth detection

# ---- Load images (validate exist) ----
for k, fname in map_images.items():
    path = os.path.join(IMAGE_FOLDER, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing image: {path}. Make sure it exists.")
    map_images[k] = path

# ---- Mediapipe setup ----
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---- Utilities ----
def normalized_to_pixel(norm_landmark, width, height):
    return int(norm_landmark.x * width), int(norm_landmark.y * height)

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_finger_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_mouth_open(face_landmarks, frame_w, frame_h):
    """Check if mouth is open using face mesh landmarks"""
    if not face_landmarks:
        return False
    
    lm = face_landmarks.landmark
    # Upper lip center: 13, Lower lip center: 14
    upper_lip = normalized_to_pixel(lm[13], frame_w, frame_h)
    lower_lip = normalized_to_pixel(lm[14], frame_w, frame_h)
    
    # Left mouth corner: 61, Right mouth corner: 291
    left_mouth = normalized_to_pixel(lm[61], frame_w, frame_h)
    right_mouth = normalized_to_pixel(lm[291], frame_w, frame_h)
    
    mouth_height = abs(upper_lip[1] - lower_lip[1])
    mouth_width = abs(left_mouth[0] - right_mouth[0])
    
    if mouth_width == 0:
        return False
    
    # Calculate aspect ratio (height/width)
    aspect_ratio = mouth_height / mouth_width
    return aspect_ratio > MOUTH_OPEN_RATIO

def get_mouth_center(face_landmarks, frame_w, frame_h):
    """Get mouth center position from face mesh"""
    if not face_landmarks:
        return (frame_w // 2, int(frame_h * 0.6))
    
    lm = face_landmarks.landmark
    # Average of upper and lower lip
    upper_lip = normalized_to_pixel(lm[13], frame_w, frame_h)
    lower_lip = normalized_to_pixel(lm[14], frame_w, frame_h)
    mouth_px = ((upper_lip[0] + lower_lip[0]) // 2, (upper_lip[1] + lower_lip[1]) // 2)
    return mouth_px

def is_index_pointing_to_mouth(hand_landmarks, mouth_px, frame_w, frame_h):
    """Check if index finger is pointing to/near mouth"""
    lm = hand_landmarks.landmark
    index_extended = is_finger_extended(lm, 8, 6)
    middle_curled = not is_finger_extended(lm, 12, 10)
    ring_curled = not is_finger_extended(lm, 16, 14)
    pinky_curled = not is_finger_extended(lm, 20, 18)

    if not (index_extended and middle_curled and ring_curled and pinky_curled):
        return False

    idx_tip_px = normalized_to_pixel(lm[8], frame_w, frame_h)
    max_dim = max(frame_w, frame_h)
    dist = euclid(idx_tip_px, mouth_px)
    return dist < (MOUTH_PROX_THRESHOLD * max_dim)

def is_index_point_up(hand_landmarks):
    """Check if index finger is pointing upward"""
    lm = hand_landmarks.landmark
    index_extended = is_finger_extended(lm, 8, 6)
    middle_curled = not is_finger_extended(lm, 12, 10)
    ring_curled = not is_finger_extended(lm, 16, 14)
    pinky_curled = not is_finger_extended(lm, 20, 18)
    
    if not (index_extended and middle_curled and ring_curled and pinky_curled):
        return False
    
    # Strict vertical check - tip should be well above pip
    return (lm[8].y + 0.05) < lm[6].y

def hands_close_together(hands_landmarks, frame_w, frame_h):
    """Check if both hands are close to each other - uses multiple reference points"""
    if len(hands_landmarks) < 2:
        return False
    
    max_dim = max(frame_w, frame_h)
    threshold = BOTH_HANDS_DIST_THRESHOLD * max_dim
    
    # Check multiple landmark pairs to detect hands meeting
    # Wrist (0), Thumb tip (4), Index tip (8), Middle tip (12), Pinky tip (20)
    check_points = [0, 4, 8, 12, 20]
    
    close_count = 0
    for point_idx in check_points:
        p0 = normalized_to_pixel(hands_landmarks[0].landmark[point_idx], frame_w, frame_h)
        p1 = normalized_to_pixel(hands_landmarks[1].landmark[point_idx], frame_w, frame_h)
        
        dist = euclid(p0, p1)
        if dist < threshold:
            close_count += 1
    
    # If at least 2 points are close, consider hands together
    return close_count >= 2

def detect_gesture_combo(hands_results, face_landmarks, frame_w, frame_h):
    """
    Detect the combination of hand gesture + mouth state.
    Returns one of: 
    - "both_hands_open_mouth": Both hands together + mouth open
    - "no_hand": No hands visible
    - "point_to_mouth": Index finger pointing at mouth (any mouth state)
    - "point_up_open_mouth": Index finger pointing up + mouth open
    - None: No matching combination
    """
    mouth_px = get_mouth_center(face_landmarks, frame_w, frame_h)
    mouth_open = is_mouth_open(face_landmarks, frame_w, frame_h)
    
    # Check hands
    if not hands_results.multi_hand_landmarks:
        return "no_hand"
    
    hands_lms = hands_results.multi_hand_landmarks
    
    # Priority 1: Both hands close together + mouth open
    if len(hands_lms) >= 2 and hands_close_together(hands_lms, frame_w, frame_h) and mouth_open:
        return "both_hands_open_mouth"
    
    # For single hand gestures, use first detected hand
    hand0 = hands_lms[0]
    
    # Priority 2: Point to mouth (finger near mouth - works with any mouth state)
    if is_index_pointing_to_mouth(hand0, mouth_px, frame_w, frame_h):
        return "point_to_mouth"
    
    # Priority 3: Point up + mouth open
    if is_index_point_up(hand0) and mouth_open:
        return "point_up_open_mouth"
    
    return None

# ---- Display helpers ----
def open_small_window(name, image_path, frame_w, frame_h):
    """Open a window the same size as the camera preview"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return None
    winname = f"MONKEY:{name}"
    
    # Resize image to match camera frame size exactly
    resized = cv2.resize(img, (frame_w, frame_h))
    
    # Create normal window (not resizable)
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winname, resized)
    return winname

def close_window(winname):
    if winname:
        try:
            cv2.destroyWindow(winname)
        except:
            pass

# ---- Main loop ----
cap = cv2.VideoCapture(0)
active_window = None
active_gesture = None
gesture_start_time = 0
last_detected_gesture = None

print("\n=== Gesture Guide ===")
print("1. Both hands together + mouth open → BothhandsWithopenmouth.png")
print("2. No hands visible → NoHandGesture.png")
print("3. Point finger to mouth → PointFingerMouth.png")
print("4. Point finger up + mouth open → PointFingerWithMoutOpen.png")
print("Press 'q' to quit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face mesh detection
        face_results = face_mesh.process(rgb)
        face_lms = None
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0]

        # Hands detection
        hands_results = hands_detector.process(rgb)
        
        # Detect gesture combination (hand + mouth)
        detected_gesture = detect_gesture_combo(hands_results, face_lms, frame_w, frame_h)

        # Gesture stability logic - only activate after holding for HOLD_TIME
        if detected_gesture != last_detected_gesture:
            gesture_start_time = time.time()
            last_detected_gesture = detected_gesture
        else:
            # Same gesture detected consistently
            if detected_gesture and time.time() - gesture_start_time >= HOLD_TIME:
                # Should we switch to this gesture?
                if active_gesture != detected_gesture:
                    # Close old window
                    if active_window:
                        close_window(active_window)
                        active_window = None
                    
                    # Open new window for the detected combo
                    if detected_gesture in map_images:
                        active_window = open_small_window(detected_gesture, map_images[detected_gesture], frame_w, frame_h)
                        active_gesture = detected_gesture
            elif not detected_gesture:
                # No valid gesture combo, close any active window
                if active_window:
                    close_window(active_window)
                    active_window = None
                    active_gesture = None

        # Draw hand landmarks with hand count indicator and distance info
        num_hands = len(hands_results.multi_hand_landmarks) if hands_results.multi_hand_landmarks else 0
        hands_together = False
        if hands_results.multi_hand_landmarks:
            hands_lms = hands_results.multi_hand_landmarks
            if len(hands_lms) >= 2:
                hands_together = hands_close_together(hands_lms, frame_w, frame_h)
                # Draw line between wrists when both hands detected
                wrist0 = normalized_to_pixel(hands_lms[0].landmark[0], frame_w, frame_h)
                wrist1 = normalized_to_pixel(hands_lms[1].landmark[0], frame_w, frame_h)
                color = (0, 255, 0) if hands_together else (0, 0, 255)
                cv2.line(frame, wrist0, wrist1, color, 2)
                cv2.putText(frame, "TOGETHER!" if hands_together else "Too Far", 
                           ((wrist0[0]+wrist1[0])//2, (wrist0[1]+wrist1[1])//2-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            for i, hlms in enumerate(hands_lms):
                mp_drawing.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
                # Draw hand number on wrist
                wrist = normalized_to_pixel(hlms.landmark[0], frame_w, frame_h)
                cv2.putText(frame, f"Hand {i+1}", (wrist[0]-30, wrist[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Visual feedback
        mouth_px = get_mouth_center(face_lms, frame_w, frame_h)
        mouth_open = is_mouth_open(face_lms, frame_w, frame_h)
        
        cv2.circle(frame, mouth_px, 8, (0, 255, 0) if mouth_open else (0, 0, 255), -1)
        
        # Status display
        num_hands = len(hands_results.multi_hand_landmarks) if hands_results.multi_hand_landmarks else 0
        status = f"Detected: {detected_gesture or 'None'} | Mouth: {'OPEN' if mouth_open else 'closed'} | Hands: {num_hands}"
        active_status = f"Active: {active_gesture or 'None'}"
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, active_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Preview (press q to quit)", frame)
        cv2.resizeWindow("Preview (press q to quit)", frame_w, frame_h)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    face_mesh.close()