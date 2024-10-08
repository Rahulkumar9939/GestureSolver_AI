import os
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import streamlit as st

from scripts.solver import Solver

solver = Solver()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Set page layout and title
st.set_page_config(layout="wide")
st.title("🤖 GestureSolver AI ✋")

# Creating columns for video feed and text display
video_col, text_col = st.columns((3, 1))

# Placeholders for video feed and text
video_placeholder = video_col.empty()
text_placeholder = text_col.empty()

# Adding some visual appeal for the solution display
text_placeholder.markdown("### **Solution Box** 🧮")
solution_box = text_placeholder.empty()

# Adding instructions on how to use the interface
st.sidebar.markdown("## 📝 Instructions")
st.sidebar.markdown("""
- Use your index and middle fingers to draw on the canvas.
- Tap **Submit** to solve the drawn equation or problem.
- Tap **Clear** to reset the canvas.
- Use the **Solution Box** to view results.
""")

# Canvas dimensions
canvas_height, canvas_width = 480, 640
canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")

# Variables for drawing and button detection
drawing = False
prev_x, prev_y = None, None
color = (0, 255, 0)

# Define button dimensions and positions
rect_width, rect_height = 100, 40
top_left_x = canvas_width - rect_width - 20  # Slight offset for visual balance
top_left_y = 20
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height

# Clear button dimensions
clear_rect_width = 100
clear_top_left_x = 20
clear_top_left_y = 20
clear_bottom_right_x = clear_top_left_x + clear_rect_width
clear_bottom_right_y = clear_top_left_y + rect_height

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2

# Button draw functions
def draw_solve_button():
    """Draws the Submit button on the canvas."""
    cv2.rectangle(canvas, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)
    text_size = cv2.getTextSize("SOLVE", font, font_scale, font_thickness)[0]
    text_x = top_left_x + (rect_width - text_size[0]) // 2
    text_y = top_left_y + (rect_height + text_size[1]) // 2
    cv2.putText(canvas, "SOLVE", (text_x, text_y), font, font_scale, font_color, font_thickness)

def draw_clear_button():
    """Draws the Clear button on the canvas."""
    cv2.rectangle(canvas, (clear_top_left_x, clear_top_left_y), (clear_bottom_right_x, clear_bottom_right_y), (255, 0, 0), -1)
    text_size = cv2.getTextSize("CLEAR", font, font_scale, font_thickness)[0]
    text_x = clear_top_left_x + (clear_rect_width - text_size[0]) // 2
    text_y = clear_top_left_y + (rect_height + text_size[1]) // 2
    cv2.putText(canvas, "CLEAR", (text_x, text_y), font, font_scale, font_color, font_thickness)

def create_canvas():
    """Creates a blank canvas with buttons."""
    global canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
    draw_solve_button()
    draw_clear_button()

create_canvas()

# Frame update function
def update_frame():
    global drawing, prev_x, prev_y, canvas

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Detecting gestures
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Check if the submit button is pressed
            if top_left_x <= x <= bottom_right_x and top_left_y <= y <= bottom_right_y:
                solution_box.markdown("### **Processing...** ⏳")
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_image_path = os.path.join(temp_dir, 'question_image_solver.png')
                    cv2.imwrite(temp_image_path, canvas)
                    result_text = solver.solve(temp_image_path)
                    solution_box.markdown(f"### **Solution**: {result_text} ✅")

            # Check if the clear button is pressed
            if clear_top_left_x <= x <= clear_bottom_right_x and clear_top_left_y <= clear_bottom_right_y:
                create_canvas()

            # Drawing lines based on gestures
            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness=5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            # Determine whether the user is drawing
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_x, middle_finger_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            if abs(x - middle_finger_x) < 40 and abs(y - middle_finger_y) < 40:
                drawing = False
            else:
                drawing = True

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame = cv2.addWeighted(frame, 1, canvas, 1.0, 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

# Initialize Mediapipe hands detection
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Main loop for updating the frame
while True:
    frame_rgb = update_frame()
    if frame_rgb.any(): 
        video_placeholder.image(frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
