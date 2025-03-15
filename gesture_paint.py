import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Set up the drawing canvas
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to open camera")
    cap.release()
    exit()

h, w, _ = frame.shape  # Get frame size
canvas = np.zeros((h, w, 3), dtype=np.uint8)
prev_x, prev_y = None, None  # Store previous index finger position
draw_color = (255, 255, 255)  # Default color is white
pen_down = True  # Track whether the pen is drawing or lifted


def detect_palm(hand_landmarks, w, h):
    """Detects if the palm is open for erasing."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    distance = abs(wrist.y * h - pinky_mcp.y * h)
    return distance > 70  # If palm is detected, return True


def get_index_finger_tip(hand_landmarks, w, h):
    """Returns the index finger tip coordinates."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return int(index_tip.x * w), int(index_tip.y * h)


def get_palm_center(hand_landmarks, w, h):
    """Estimate palm center as the midpoint of the wrist and middle finger MCP joint."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    palm_x = int((wrist.x + middle_mcp.x) / 2 * w)
    palm_y = int((wrist.y + middle_mcp.y) / 2 * h)
    return palm_x, palm_y

# Create color palette
color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
palette_h = 50
palette = np.zeros((palette_h, w, 3), dtype=np.uint8)
for i, color in enumerate(color_palette):
    cv2.rectangle(palette, (i * (w // len(color_palette)), 0), ((i + 1) * (w // len(color_palette)), palette_h), color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  # Consider only one hand
        
        if detect_palm(hand_landmarks, w, h):
            # Erase using palm position
            palm_x, palm_y = get_palm_center(hand_landmarks, w, h)
            erase_radius = 50  # Increase this value for larger erasing area

            # Create a mask where the palm is, and erase only that area
            mask = np.ones_like(canvas, dtype=np.uint8) * 255
            cv2.circle(mask, (palm_x, palm_y), erase_radius, (0, 0, 0), -1)
            canvas = cv2.bitwise_and(canvas, mask)  # Properly erase the area
        else:
            pen_down = True  # Pen down (drawing mode)
            x, y = get_index_finger_tip(hand_landmarks, w, h)
            
            # Check if index finger is in color palette area
            if y < palette_h:
                color_index = x // (w // len(color_palette))
                draw_color = color_palette[color_index]
            elif pen_down:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
                prev_x, prev_y = x, y  # Update previous position
                
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None  # Reset when no hand is detected
    
    # Resize canvas to match frame size before blending
    canvas_resized = cv2.resize(canvas, (w, h))
    output = cv2.addWeighted(frame, 0.5, canvas_resized, 0.5, 0)
    
    # Add color palette on top
    output[:palette_h, :] = palette
    
    cv2.imshow("Writing Pad", output)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
