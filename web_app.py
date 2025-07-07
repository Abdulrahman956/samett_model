import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
import tensorflow as tf
import copy
import csv
import time

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Load the pre-trained model
keypoint_classifier = tf.lite.Interpreter(model_path='model/keypoint_classifier/keypoint_classifier.tflite')
keypoint_classifier.allocate_tensors()

# Print model input details for debugging
print("Keypoint Classifier Input Details:", keypoint_classifier.get_input_details())

# Load labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# Global variables
camera = None
streaming = False
detected_signs = []  # Store detected signs

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    print("Raw landmark_list length:", len(landmark_point))
    if not landmark_point:
        print("Warning: landmark_point is empty!")
    return landmark_point

def pre_process_landmark(landmark_list):
    print("Input landmark_list length:", len(landmark_list))
    if not landmark_list:
        print("Warning: landmark_list is empty in pre_process_landmark!")
        return []
    temp_landmark_list = copy.deepcopy(landmark_list)
    for lm in temp_landmark_list:
        if not isinstance(lm, list) or len(lm) != 2 or not all(isinstance(v, (int, float)) for v in lm):
            print("Warning: Invalid landmark data:", lm)
            return []
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = np.array(temp_landmark_list).flatten()
    max_value = np.max(np.abs(temp_landmark_list))
    if max_value == 0:
        print("Warning: max_value is 0, skipping normalization")
        return temp_landmark_list.tolist()
    temp_landmark_list = temp_landmark_list / max_value
    print("Preprocessed landmark_list shape:", temp_landmark_list.shape)
    if temp_landmark_list.size == 0:
        print("Warning: preprocessed landmark_list is empty!")
    return temp_landmark_list.tolist()

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        for index, landmark in enumerate(landmark_point):
            if index in [4, 8, 12, 16, 20]:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    return image

def generate_frames():
    global camera, streaming, detected_signs
    last_sign = None
    last_sign_time = 0
    sign_duration = 1.0  # Minimum time (seconds) to consider a sign stable
    while True:
        if camera is None:
            print("Initializing camera in generate_frames")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Could not open camera in generate_frames")
                return
        while streaming:
            success, image = camera.read()
            if not success:
                print("Error: Failed to read frame from camera")
                break
            print("Frame read successfully")
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[hand_idx]
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    landmark_list = landmark_list[:21]
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    if not pre_processed_landmark_list:
                        print(f"Skipping model inference for hand {hand_idx}: pre_processed_landmark_list is empty")
                        continue

                    keypoint_classifier.set_tensor(
                        keypoint_classifier.get_input_details()[0]['index'],
                        np.array([pre_processed_landmark_list], dtype=np.float32))
                    keypoint_classifier.invoke()
                    hand_sign_id = np.argmax(keypoint_classifier.get_tensor(
                        keypoint_classifier.get_output_details()[0]['index'])[0])
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id]

                    # Add stable signs to detected_signs
                    current_time = time.time()
                    if hand_sign_text != last_sign:
                        last_sign = hand_sign_text
                        last_sign_time = current_time
                    elif (current_time - last_sign_time) >= sign_duration and hand_sign_text not in detected_signs[-1:]:
                        detected_signs.append(hand_sign_text)
                        if len(detected_signs) > 10:  # Limit to last 10 signs
                            detected_signs.pop(0)

                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    # Removed draw_info_text to avoid displaying result above hand
            else:
                debug_image = copy.deepcopy(image)  # No text when no hands detected

            ret, buffer = cv2.imencode('.jpg', debug_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        print("Streaming stopped, waiting for restart")
        if camera is not None:
            camera.release()
            camera = None
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streaming, camera
    if not streaming:
        streaming = True
        if camera is None or not camera.isOpened():
            print("Starting stream: Initializing camera")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Could not open camera in start_stream. Trying alternative index (1)...")
                camera = cv2.VideoCapture(1)
                if not camera.isOpened():
                    print("Error: Could not open any camera")
                    streaming = False
                    return "Error: Could not open camera", 500
            print("Camera opened successfully with index 0") if camera.get(cv2.CAP_PROP_POS_FRAMES) == 0 else print("Camera reopened successfully")
        print("Stream started successfully")
    return '', 204

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global streaming, camera, detected_signs
    streaming = False
    detected_signs = []  # Clear detected signs on stop
    if camera is not None:
        print("Stopping stream: Releasing camera")
        camera.release()
        camera = None
        time.sleep(1)
        print("Camera released and reset")
    print("Stream stopped successfully")
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_signs')
def get_signs():
    global detected_signs
    return jsonify({'signs': detected_signs})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  #192.168.1.4