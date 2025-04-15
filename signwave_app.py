import cv2
import numpy as np
import threading
import tkinter as tk
import time
from tkinter import Label, Button, Frame, font
from PIL import Image, ImageTk
import tensorflow as tf
import mediapipe as mp
import os
import sys  # Added for resource_path
import traceback # For detailed error printing

# --- Helper function for PyInstaller ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # sys._MEIPASS is the standard attribute name.
        base_path = sys._MEIPASS
        print(f"Running bundled app, base path: {base_path}")
    except Exception:
        # Not running in a bundle (development mode)
        # Use the directory of the script file
        base_path = os.path.abspath(os.path.dirname(__file__))
        print(f"Running in dev mode, base path: {base_path}")

    return os.path.join(base_path, relative_path)
# --- End Helper function ---

# Initialize MediaPipe Hand solution
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, # Process only one hand
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe Hands initialized successfully.")
except Exception as e:
    print(f"Error initializing MediaPipe Hands: {e}")
    hands = None # Set to None if initialization fails

# Alphabet list (ensure this matches your model's output classes)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXY'

# Global variables
current_prediction = "None"
prediction_active = True
debug_mode = False # Set to False for release builds
model = None
hand_detected_time = None
stabilization_period = 5  # seconds
is_stabilizing = False
hand_roi = None # Stores the region of interest for saving samples

# Status button references (initialized later)
hand_status_btn = None
pred_status_btn = None
stab_status_btn = None

# --- Load the Model ---
try:
    model_filename = "signlanguagemodel.h5"
    model_path = resource_path(model_filename)
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model file not found at expected location: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    # model.summary() # Optional: Print model summary during debug
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
    print(traceback.format_exc())
    # Optionally, display error in GUI if Tkinter is already initialized
    # For simplicity here, we just print and model remains None

# --- Define Sample Directories ---
# NOTE: These folders will be created in the directory where the executable is run.
# Ensure the user has write permissions in that location (e.g., Desktop, Downloads).
training_samples_dir = "training_samples"
incorrect_samples_dir = "incorrect_samples"
try:
    os.makedirs(training_samples_dir, exist_ok=True)
    os.makedirs(incorrect_samples_dir, exist_ok=True)
    print(f"Sample directories ensured at: {os.path.abspath(training_samples_dir)} and {os.path.abspath(incorrect_samples_dir)}")
except OSError as e:
    print(f"Warning: Could not create sample directories: {e}. Saving samples might fail.")
    # Optionally disable sample saving features if directories can't be made

# --- Preprocessing Function ---
def preprocess_image(img_input):
    """ Preprocesses the hand ROI image for model prediction. """
    try:
        if img_input is None or img_input.size == 0:
            print("Preprocessing error: Input image is empty.")
            return None

        # Ensure image is in BGR format if it has 3 channels initially
        if len(img_input.shape) == 3 and img_input.shape[2] == 3:
            img = img_input
        elif len(img_input.shape) == 2: # Grayscale input
             img = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR) # Convert to BGR first
        else:
            print(f"Preprocessing error: Unexpected image shape {img_input.shape}")
            return None

        # Resize to model's expected input size (e.g., 200x200)
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Image Enhancement Steps ---
        # Histogram Equalization
        equalized_img = cv2.equalizeHist(gray_img)

        # Denoising (choose one or combine carefully)
        denoised_img = cv2.medianBlur(equalized_img, 3)
        # denoised_img = cv2.fastNlMeansDenoising(equalized_img, None, 10, 7, 21) # Can be slow

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(denoised_img)

        # Gaussian Blur (gentle smoothing)
        blurred_img = cv2.GaussianBlur(clahe_img, (3, 3), 0)
        # --- End Enhancement ---

        # Convert back to 3 channels (RGB) as many CNNs expect this
        # If your model was trained on grayscale, skip this and adjust normalization/dims
        final_img_gray = blurred_img # Use the enhanced grayscale result
        final_img_rgb = cv2.cvtColor(final_img_gray, cv2.COLOR_GRAY2RGB)

        # Normalize pixel values to [0, 1]
        normalized_img = final_img_rgb.astype(np.float32) / 255.0

        # Expand dimensions to add batch size (1, height, width, channels)
        processed_img = np.expand_dims(normalized_img, axis=0)

        return processed_img

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        print(traceback.format_exc())
        return None

# --- GUI Action Functions ---
def toggle_prediction():
    global prediction_active, hand_detected_time, is_stabilizing
    prediction_active = not prediction_active
    status_text = "Pause Prediction" if prediction_active else "Resume Prediction"
    toggle_btn.config(text=status_text)

    if not prediction_active:
        label_text.set("Prediction Paused")
        # Reset stabilization state when pausing
        hand_detected_time = None
        is_stabilizing = False
        current_prediction = "None" # Reset prediction display
    else:
         label_text.set("Initializing...") # Or previous state if applicable

    update_status_buttons()

def collect_sample():
    """ Saves the current hand_roi as a training sample. """
    global current_prediction, hand_roi, sample_label
    if hand_roi is None or hand_roi.size == 0:
        sample_label.config(text="No hand image to save.")
        print("Collect Sample: No hand_roi available.")
        return
    if current_prediction == "None":
        sample_label.config(text="No prediction to associate sample with.")
        print("Collect Sample: No current prediction.")
        return

    try:
        label_dir = os.path.join(training_samples_dir, current_prediction)
        os.makedirs(label_dir, exist_ok=True) # Ensure subdirectory exists

        timestamp = int(time.time() * 1000) # Millisecond timestamp for uniqueness
        filename = os.path.join(label_dir, f"train_{timestamp}.jpg")

        # Save the captured hand_roi (should be BGR)
        success = cv2.imwrite(filename, hand_roi)
        if success:
            feedback = f"Saved sample for {current_prediction}"
            print(f"Saved training sample to {filename}")
        else:
            feedback = f"Failed to save sample for {current_prediction}"
            print(f"Failed to write training sample to {filename}")
        sample_label.config(text=feedback)

    except OSError as e:
         print(f"Error creating directory or saving training sample: {e}")
         sample_label.config(text="Error saving sample (OS Error)")
    except Exception as e:
        print(f"Unexpected error saving training sample: {e}")
        print(traceback.format_exc())
        sample_label.config(text="Error saving sample")


def report_incorrect():
    """ Saves the current hand_roi as an incorrect sample. """
    global current_prediction, hand_roi, sample_label
    if hand_roi is None or hand_roi.size == 0:
        sample_label.config(text="No hand image to report.")
        print("Report Incorrect: No hand_roi available.")
        return
    if current_prediction == "None":
        sample_label.config(text="No prediction made to report as incorrect.")
        print("Report Incorrect: No current prediction.")
        return

    try:
        label_dir = os.path.join(incorrect_samples_dir, current_prediction)
        os.makedirs(label_dir, exist_ok=True) # Ensure subdirectory exists

        timestamp = int(time.time() * 1000)
        filename = os.path.join(label_dir, f"incorrect_{timestamp}.jpg")

        # Save the captured hand_roi (should be BGR)
        success = cv2.imwrite(filename, hand_roi)
        if success:
            feedback = f"Reported incorrect: {current_prediction}"
            print(f"Saved incorrect sample to {filename}")
        else:
            feedback = f"Failed to report incorrect: {current_prediction}"
            print(f"Failed to write incorrect sample to {filename}")
        sample_label.config(text=feedback)

    except OSError as e:
         print(f"Error creating directory or saving incorrect sample: {e}")
         sample_label.config(text="Error reporting (OS Error)")
    except Exception as e:
        print(f"Unexpected error saving incorrect sample: {e}")
        print(traceback.format_exc())
        sample_label.config(text="Error reporting sample")

def end_application():
    """ Stops the video stream and closes the Tkinter window. """
    print("End application requested.")
    global cap, root, hands
    # Stop the video loop if it's running (optional, mainloop quit handles it)
    if cap and cap.isOpened():
        cap.release()
        print("Camera released.")
    if hands:
        try:
            hands.close()
            print("MediaPipe Hands closed.")
        except Exception as e:
            print(f"Error closing MediaPipe hands: {e}")
    if root:
        root.quit()     # Stop the Tkinter main loop
        root.destroy()  # Destroy the Tkinter window
        print("Tkinter window destroyed.")

# --- Status Update Function ---
def update_status_buttons():
    """ Updates the color and text of status indicator buttons. """
    global hand_status_btn, pred_status_btn, stab_status_btn
    global hand_detected_time, is_stabilizing, prediction_active

    # Ensure buttons are initialized before updating
    if not all([hand_status_btn, pred_status_btn, stab_status_btn]):
        return

    # Hand Detection Status
    hand_detected = hand_detected_time is not None
    hand_bg = "#00cc00" if hand_detected else "#cc0000" # Brighter green/red
    hand_text = "Hand: ON" if hand_detected else "Hand: OFF"
    hand_status_btn.config(bg=hand_bg, text=hand_text)

    # Prediction Status
    pred_bg = "#00cc00" if prediction_active else "#cc0000"
    pred_text = "Pred: ON" if prediction_active else "Pred: OFF"
    pred_status_btn.config(bg=pred_bg, text=pred_text)

    # Stabilization Status
    stab_bg = "#ffff00" if is_stabilizing else "#a0a0a0" # Yellow / Gray
    stab_text = "Stab: ACTIVE" if is_stabilizing else "Stab: IDLE"
    stab_status_btn.config(bg=stab_bg, text=stab_text)


# --- Main Video Processing and Prediction Function ---
def video_stream():
    """ Captures frame, detects hand, preprocesses, predicts, and updates GUI. """
    global cap, lbl, label_text, current_prediction, hand_roi
    global hand_detected_time, is_stabilizing, model, hands

    # Crucial checks before processing
    if model is None:
        label_text.set("FATAL: Model not loaded!")
        update_status_buttons() # Show status even if broken
        root.after(1000, video_stream) # Retry check later
        return
    if hands is None:
        label_text.set("FATAL: MediaPipe not loaded!")
        update_status_buttons()
        root.after(1000, video_stream)
        return
    if not cap or not cap.isOpened():
        label_text.set("ERROR: Cannot access camera!")
        update_status_buttons()
        root.after(1000, video_stream)
        return

    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame.")
        label_text.set("Camera Frame Error!")
        # Keep trying to read frames
        root.after(50, video_stream) # Try again sooner
        return

    # --- Frame Processing ---
    frame = cv2.flip(frame, 1) # Mirror effect
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False # Performance optimization

    # Process with MediaPipe Hands
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True # Make writeable again

    hand_detected_this_frame = False
    hand_roi_local = None # Use local ROI for processing this frame

    if results.multi_hand_landmarks:
        # --- Hand Detected ---
        hand_detected_this_frame = True
        hand_landmarks = results.multi_hand_landmarks[0] # Get first hand

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate bounding box
        x_coords = [landmark.x * frame_width for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * frame_height for landmark in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        padding = 30 # Increased padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Green box

        # Extract ROI if valid
        if x_max > x_min and y_max > y_min:
            hand_roi_local = frame[y_min:y_max, x_min:x_max].copy() # Use copy!
            # Update global hand_roi primarily for saving samples later
            hand_roi = hand_roi_local

        # --- Stabilization Logic ---
        current_time = time.time()
        if hand_detected_time is None: # Start timer if hand just appeared
            hand_detected_time = current_time
            is_stabilizing = True
            label_text.set(f"Stabilizing: Hold for {stabilization_period}s")

        elapsed_time = current_time - hand_detected_time
        remaining_time = max(0, stabilization_period - elapsed_time)

        if is_stabilizing:
            label_text.set(f"Stabilizing: {remaining_time:.1f}s")
            cv2.putText(frame, f"Hold: {remaining_time:.1f}s", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text

            if elapsed_time >= stabilization_period:
                is_stabilizing = False # Stabilization complete

                # --- Prediction Trigger ---
                if prediction_active and hand_roi_local is not None:
                    processed_frame = preprocess_image(hand_roi_local)
                    if processed_frame is not None:
                        try:
                            prediction = model.predict(processed_frame)[0] # Get first batch result
                            predicted_label_idx = np.argmax(prediction)
                            confidence = prediction[predicted_label_idx]
                            threshold = 0.6 # Confidence threshold

                            if predicted_label_idx < len(alphabet):
                                predicted_letter = alphabet[predicted_label_idx]
                                current_prediction = predicted_letter # Update global prediction

                                if confidence >= threshold:
                                    label_text.set(f"Predicted: {predicted_letter} ({confidence:.2f})")
                                else:
                                    label_text.set(f"Low Conf: {predicted_letter} ({confidence:.2f})")

                                if debug_mode:
                                     print(f"Prediction: {predicted_letter}, Conf: {confidence:.4f}")
                                     # Print top 3 predictions
                                     top_indices = np.argsort(prediction)[::-1][:3]
                                     top_preds = [(alphabet[i], prediction[i]) for i in top_indices if i < len(alphabet)]
                                     print(f"Top 3: {top_preds}")

                            else:
                                print(f"Error: Predicted index {predicted_label_idx} out of bounds.")
                                label_text.set("Prediction Index Error")
                                current_prediction = "None"

                        except Exception as e:
                            print(f"Error during prediction: {e}")
                            print(traceback.format_exc())
                            label_text.set("Prediction Failed")
                            current_prediction = "None"
                    else:
                        label_text.set("Preprocessing Failed")
                        current_prediction = "None"
                elif not prediction_active:
                     label_text.set("Prediction Paused") # Keep showing paused if hand is stable
                     current_prediction = "None"
                     # Don't reset timer here, hand is still stable

        # --- End Stabilization Logic ---

    # --- Hand Not Detected ---
    if not hand_detected_this_frame:
        if hand_detected_time is not None: # Hand was just lost
            print("Hand lost.")
        hand_detected_time = None # Reset timer
        is_stabilizing = False # Stop stabilizing
        hand_roi = None # Clear saved ROI
        current_prediction = "None" # Clear prediction
        if prediction_active:
             label_text.set("No Hand Detected") # Update label only if predicting

    # --- Update GUI ---
    update_status_buttons() # Update indicators regardless of detection

    try:
        # Convert frame for Tkinter display
        frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb_display)
        img = img.resize((600, 450), Image.Resampling.LANCZOS) # Resize for display
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk # Keep reference
        lbl.configure(image=imgtk)
    except Exception as e:
        print(f"Error updating Tkinter image: {e}")
        # This might happen if the window is closing

    # Schedule the next frame processing
    if root: # Check if root window still exists
        root.after(15, video_stream) # ~66 FPS target, adjust as needed

# --- Initialize Tkinter GUI ---
print("Initializing Tkinter GUI...")
root = tk.Tk()
root.title("SignWave - Sign Language Recognition")
root.geometry("1200x700") # Adjust size as needed
root.configure(bg="#f0f0f0")
root.protocol("WM_DELETE_WINDOW", end_application) # Handle window close button

# Header
header_frame = Frame(root, bg="#1a56c4", height=60)
header_frame.pack(fill=tk.X)
header_frame.pack_propagate(False)
Label(header_frame, text="SignWave", font=("Arial", 24, "bold"), bg="#1a56c4", fg="white").place(x=20, y=10)
Label(header_frame, text="Sign Language Recognition", font=("Arial", 16), bg="#1a56c4", fg="white").place(x=200, y=18)

# Main Content Area
content_frame = Frame(root, bg="#f0f0f0")
content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# --- Left Panel (Instructions & Status) ---
left_panel = Frame(content_frame, bg="white", bd=1, relief=tk.SOLID, width=450) # Adjusted width
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10)) # Fill Y only
left_panel.pack_propagate(False)

# Instructions Section
instructions_header = Frame(left_panel, bg="#2a68d6", height=40)
instructions_header.pack(fill=tk.X)
Label(instructions_header, text="How to Use SignWave", font=("Arial", 14, "bold"), bg="#2a68d6", fg="white").pack(pady=8)

instructions_content_frame = Frame(left_panel, bg="white")
instructions_content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

instructions = [
    "Position your hand clearly in the camera view.",
    "Ensure good, consistent lighting.",
    "Keep a comfortable distance from the camera.",
    "Perform the sign language gesture.",
    "Hold your hand steady for 5 seconds for stabilization.",
    "The predicted sign will appear below the camera feed.",
    "If the prediction is incorrect, click 'INCORRECT'.",
    "Click 'END' or press 'Esc'/'q' to close."
]
for i, text in enumerate(instructions):
    f = Frame(instructions_content_frame, bg="white")
    f.pack(fill=tk.X, pady=4)
    Label(f, text="•", font=("Arial", 14, "bold"), fg="#2a68d6", bg="white").pack(side=tk.LEFT, padx=(0, 5))
    Label(f, text=text, font=("Arial", 11), justify=tk.LEFT, bg="white", anchor="w", wraplength=380).pack(side=tk.LEFT, fill=tk.X)

# Status Indicators Section
status_frame = Frame(left_panel, bg="white", bd=0)
status_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
Label(status_frame, text="Status Indicators", font=("Arial", 13, "bold"), bg="white", fg="#2a68d6").pack(anchor=tk.W, pady=(0, 5))

status_buttons_frame = Frame(status_frame, bg="white")
status_buttons_frame.pack(fill=tk.X, pady=5)
button_style = {"font": ("Arial", 9, "bold"), "width": 10, "height": 1, "relief": tk.RAISED, "bd": 2, "fg": "white", "state": tk.DISABLED} # Disabled style

hand_status_btn = Button(status_buttons_frame, text="Hand: -", bg="#a0a0a0", **button_style); hand_status_btn.pack(side=tk.LEFT, padx=5)
pred_status_btn = Button(status_buttons_frame, text="Pred: -", bg="#a0a0a0", **button_style); pred_status_btn.pack(side=tk.LEFT, padx=5)
stab_status_btn = Button(status_buttons_frame, text="Stab: -", bg="#a0a0a0", **button_style); stab_status_btn.pack(side=tk.LEFT, padx=5)
Label(status_frame, text="Real-time system status (not clickable).", font=("Arial", 9, "italic"), fg="#555555", bg="white").pack(anchor=tk.W, padx=5, pady=(5, 10))

# --- Right Panel (Camera & Controls) ---
right_panel = Frame(content_frame, bg="white", bd=1, relief=tk.SOLID)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Camera Feed Section
camera_header = Frame(right_panel, bg="#2a68d6", height=40)
camera_header.pack(fill=tk.X)
Label(camera_header, text="Camera Feed", font=("Arial", 14, "bold"), bg="#2a68d6", fg="white").pack(pady=8)

camera_display_frame = Frame(right_panel, bg="black", padx=5, pady=5) # Added frame for padding
camera_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
lbl = Label(camera_display_frame, bg="black") # Video display label
lbl.pack(fill=tk.BOTH, expand=True)

# Prediction Output Section
prediction_frame = Frame(right_panel, bg="white", height=90) # Slightly taller
prediction_frame.pack(fill=tk.X, padx=10, pady=5)
prediction_frame.pack_propagate(False)
Label(prediction_frame, text="Predicted Output:", font=("Arial", 13, "bold"), bg="white", fg="#2a68d6").pack(anchor=tk.W, pady=(5, 2))

label_text = tk.StringVar(); label_text.set("Initializing...") # For prediction display
label = Label(prediction_frame, textvariable=label_text, font=("Arial", 18, "bold"), bg="#e0e0e0", relief=tk.GROOVE, height=1, anchor='w', padx=10)
label.pack(fill=tk.X, pady=(0, 5))

sample_label = Label(prediction_frame, text="", font=("Arial", 10, "italic"), bg="white", fg="#2a68d6") # Feedback label
sample_label.pack(anchor=tk.W, pady=(0, 5))

# Buttons Section
button_frame = Frame(right_panel, bg="white", padx=10)
button_frame.pack(fill=tk.X,pady=(5, 10))
btn_font = ("Arial", 11, "bold"); btn_width = 15; btn_height = 2; btn_padx = 12

toggle_btn = Button(button_frame, text="Pause Prediction", font=btn_font, width=btn_width, height=btn_height, bg="#3498db", fg="white", relief=tk.RAISED, bd=3, command=toggle_prediction)
toggle_btn.pack(side=tk.LEFT, padx=btn_padx, pady=5)

# sample_btn = Button(button_frame, text="Save Sample", font=btn_font, width=btn_width, height=btn_height, bg="#2ecc71", fg="white", relief=tk.RAISED, bd=3, command=collect_sample)
# sample_btn.pack(side=tk.LEFT, padx=btn_padx, pady=5) # Uncomment if needed

incorrect_btn = Button(button_frame, text="INCORRECT", font=btn_font, width=btn_width, height=btn_height, bg="#e67e22", fg="white", relief=tk.RAISED, bd=3, command=report_incorrect)
incorrect_btn.pack(side=tk.LEFT, padx=btn_padx, pady=5)

end_btn = Button(button_frame, text="END", font=btn_font, width=btn_width, height=btn_height, bg="#e74c3c", fg="white", relief=tk.RAISED, bd=3, command=end_application)
end_btn.pack(side=tk.LEFT, padx=btn_padx, pady=5)

# --- Initialize Camera ---
print("Initializing camera...")
cap = cv2.VideoCapture(0) # Try camera index 0 first
if not cap or not cap.isOpened():
    print("Warning: Could not open camera 0. Trying camera 1...")
    cap = cv2.VideoCapture(1) # Try camera index 1 as fallback

if not cap or not cap.isOpened():
    print("FATAL ERROR: Cannot access any webcam!")
    label_text.set("FATAL ERROR: Cannot access webcam!")
    # Keep GUI running but indicate the error
else:
    print(f"Camera opened successfully (index used: {cap.get(cv2.CAP_PROP_POS_FRAMES)})") # Check which index worked

# --- Bind Keys ---
root.bind("<space>", lambda event: toggle_prediction())
# root.bind("<s>", lambda event: collect_sample()) # Uncomment if sample button is active
root.bind("<i>", lambda event: report_incorrect())
root.bind("<q>", lambda event: end_application())
root.bind("<Escape>", lambda event: end_application())

# --- Start Processes ---
update_status_buttons() # Initial status update


    # <<< --- ADD DETAILED CHECK HERE --- >>>
print("\n--- Status Check Before Starting Stream ---")

try:
    cap_ok = cap is not None and cap.isOpened()
    print(f"Camera object exists: {cap is not None}")
    print(f"Camera is opened: {cap.isOpened() if cap else 'N/A'}")
except Exception as e:
    cap_ok = False
    print(f"Error checking camera status: {e}")

model_ok = model is not None
hands_ok = hands is not None
print(f"Model object exists: {model_ok}")
print(f"MediaPipe Hands object exists: {hands_ok}")
print("-----------------------------------------\n")
# <<< --- END OF DETAILED CHECK --- >>>


# Use the checked variables in the condition
if cap_ok and model_ok and hands_ok:
    print("All checks passed. Starting video stream...") # Modified print
    video_stream() # Start the main loop
else:
    print("Video stream not started due to failed checks.") # Modified print
    # Add specific error messages
    error_msg = "ERROR:"
    if not cap_ok: error_msg += " Camera Not Ready!"
    if not model_ok: error_msg += " Model Not Loaded!"
    if not hands_ok: error_msg += " MediaPipe Not Loaded!"
    # Add a fallback for safety, though unlikely needed now
    if error_msg == "ERROR:": error_msg = "ERROR: Unknown Initialization Issue!"
    print(f"Failure Reason: {error_msg}") # Print specific reason
    label_text.set(error_msg) # Show specific error in GUI

print("Starting Tkinter main loop...")
root.mainloop()



# --- Cleanup (after mainloop ends) ---
print("Application exited.")
# Resources should be released in end_application() or when video_stream stops