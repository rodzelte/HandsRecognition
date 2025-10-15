# Gesture Recognition System with Angle Detection

A real-time hand gesture and facial expression recognition system using MediaPipe and OpenCV. This system detects various hand gestures combined with mouth states and displays corresponding images.

## Features

- **Angle-Based Detection**: Robust pointing detection that works from any angle, including side views
- **Real-time Processing**: Fast gesture recognition using MediaPipe
- **Multiple Gesture Combinations**:
  - Both hands together + mouth open
  - No hands visible
  - Index finger pointing to mouth (works from any angle)
  - Index finger pointing up + mouth open
- **Stable Detection**: Gestures must be held for 0.4 seconds to prevent false positives
- **Visual Feedback**: Real-time visualization of hand landmarks, pointing direction, and mouth status

## Requirements

- Python 3.7 or higher
- Webcam

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install Dependencies

Run the following command to install all required packages:

```bash
pip install opencv-python mediapipe numpy
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Detailed Package Information

- **opencv-python** (4.8.0+): Computer vision library for video capture and image processing
- **mediapipe** (0.10.0+): Google's ML framework for hand and face landmark detection
- **numpy** (1.24.0+): Numerical computing library for angle calculations

## Project Structure

```
project-folder/
├── gesture_recognition.py    # Main script
├── Images/                    # Image folder (create this)
│   ├── BothhandsWithopenmouth.png
│   ├── NoHandGesture.png
│   ├── PointFingerMouth.png
│   └── PointFingerWithMoutOpen.png
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### 1. Create the Images Folder

Create a folder named `Images` in the same directory as the script:

```bash
mkdir Images
```

### 2. Add Gesture Images

Place the following images in the `Images` folder:
- `BothhandsWithopenmouth.png` - Displayed when both hands are together with mouth open
- `NoHandGesture.png` - Displayed when no hands are detected
- `PointFingerMouth.png` - Displayed when pointing finger to mouth
- `PointFingerWithMoutOpen.png` - Displayed when pointing finger up with mouth open

**Note**: The script will raise an error if any image is missing.

## Usage

### Running the Script

```bash
python gesture_recognition.py
```

### Gestures

1. **Both Hands Together + Mouth Open**
   - Bring both hands close together (any part of the hands)
   - Open your mouth
   - Hold for 0.4 seconds

2. **No Hands**
   - Keep your hands out of the camera frame
   - Image displays after 0.4 seconds

3. **Point to Mouth** (Works from any angle!)
   - Extend your index finger
   - Curl your other fingers (middle, ring, pinky)
   - Point your finger toward your mouth
   - Works from front view, side view, and any angle in between

4. **Point Up + Mouth Open**
   - Extend your index finger pointing upward
   - Curl your other fingers
   - Open your mouth
   - Hold for 0.4 seconds

### Controls

- Press **'q'** to quit the application

### Visual Indicators

- **Green circle** on mouth: Mouth is open
- **Red circle** on mouth: Mouth is closed
- **Purple line**: Shows pointing direction for debugging
- **Yellow dot**: Index finger tip
- **Hand labels**: Shows "Hand 1" and "Hand 2" when detected
- **Green/Red line** between hands: Green = hands together, Red = too far apart

## Configuration

You can adjust the sensitivity by modifying these constants in the script:

```python
HOLD_TIME = 0.4                    # Time to hold gesture (seconds)
MOUTH_PROX_THRESHOLD = 0.12        # Distance threshold for pointing to mouth
BOTH_HANDS_DIST_THRESHOLD = 0.45   # Distance threshold for hands together
MOUTH_OPEN_RATIO = 0.12            # Mouth aspect ratio for open detection
POINTING_ANGLE_THRESHOLD = 35      # Max angle (degrees) for pointing detection
```

## Troubleshooting

### Camera Not Found
- Ensure your webcam is connected and not being used by another application
- Try changing `cap = cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher if you have multiple cameras

### Images Not Loading
- Verify all image files are in the `Images` folder
- Check that filenames match exactly (case-sensitive)
- Ensure images are valid PNG files

### Gesture Not Detected
- Ensure good lighting conditions
- Keep your hand visible and not too close to the camera
- Try adjusting the configuration constants
- Check the purple pointing line for debugging direction

### Side View Not Working
- Make sure you're making a clear pointing gesture (index extended, others curled)
- The purple line should point toward your target
- Adjust `POINTING_ANGLE_THRESHOLD` if needed (higher = more lenient)

## Technical Details

### Angle-Based Detection Algorithm

The system uses vector mathematics to calculate the angle between:
1. **Finger Direction**: Vector from index finger base (MCP joint) to tip
2. **Target Direction**: Vector from finger tip to target (mouth or upward point)

This approach is robust to hand orientation and works from any viewing angle, including side views.

### Landmark Indices Used

- **Hand**: Wrist (0), Index MCP (5), Index Tip (8), Other finger tips for gesture validation
- **Face**: Nose tip (1), Upper lip (13), Lower lip (14), Mouth corners (61, 291)

## Performance Tips

- Keep distance from camera between 0.5-2 meters
- Ensure good, even lighting
- Avoid cluttered backgrounds
- Make clear, distinct gestures

## License

Open Project

## Contributing

[Add contribution guidelines here]

## Author

Rodzel Te

## Acknowledgments

- Built with [MediaPipe](https://mediapipe.dev/) by Google
- Uses [OpenCV](https://opencv.org/) for video processing
