import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import tensorflow as tf
from collections import deque

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Risk factor mapping (Bias score calculation)
risk_factors = {
    'person': 1.0, 'car': 0.8, 'bicycle': 0.6, 'motorbike': 0.7,
    'bus': 0.75, 'truck': 0.85, 'dog': 0.9, 'cat': 0.6,
}

# Normalize weights
total_weight = sum(risk_factors.values())
normalized_weights = {obj: risk_factors[obj] / total_weight for obj in risk_factors}

# Ethical decision-making factor mapping
ethical_factors = {
    'person': 1.2,  # Higher priority for person detection
    'car': 0.8,     # Lower priority for car detection (for example)
    'bicycle': 0.9, # Moderate priority for bicycle
    'motorbike': 1.0,
    'bus': 0.7,
    'truck': 0.7,
    'dog': 0.6,     # Animals may have lower ethical priority
    'cat': 0.6,
}

# Normalize ethical weights
total_ethical_weight = sum(ethical_factors.values())
normalized_ethical_weights = {obj: ethical_factors[obj] / total_ethical_weight for obj in ethical_factors}

# Directory containing images
data_dir = "all_data/"
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Ground truth (input by the user)
ground_truth = {}

def preprocess_image(img_path):
    """Load and preprocess image."""
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    return img, blob, height, width

def detect_objects_in_image(img, blob, height, width):
    """Detect objects in the image using YOLO and apply NMS."""
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Lowered threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression (NMS) to reduce duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)  # Confidence threshold, NMS threshold

    # Filter out the boxes that were suppressed
    filtered_class_ids = [class_ids[i] for i in indices.flatten()]
    filtered_confidences = [confidences[i] for i in indices.flatten()]
    filtered_boxes = [boxes[i] for i in indices.flatten()]

    return filtered_class_ids, filtered_confidences, filtered_boxes

def calculate_bias_score(class_ids):
    """Calculate the bias score based on detected objects."""
    bias_score = 0
    detected_objects = []
    for class_id in class_ids:
        class_name = classes[class_id]  # Map class_id to actual class name
        if class_name in normalized_weights:
            bias_score += normalized_weights[class_name]
            detected_objects.append(class_name)
    return bias_score, detected_objects

def calculate_ethical_score(class_ids):
    """Calculate the ethical decision-making score based on detected objects."""
    ethical_score = 0
    detected_objects = []
    
    for class_id in class_ids:
        class_name = classes[class_id]  # Map class_id to actual class name
        if class_name in normalized_ethical_weights:
            ethical_score += normalized_ethical_weights[class_name]
            detected_objects.append(class_name)
    
    # Only calculate ethical score for detected objects
    return ethical_score, detected_objects


def calculate_fnr_fps(true_objects, detected_objects):
    """Calculate the False Negative Rate (FNR) and False Positive Score (FPS)."""
    true_positives = len([obj for obj in detected_objects if obj in true_objects])
    false_negatives = len(true_objects) - true_positives
    false_positives = len([obj for obj in detected_objects if obj not in true_objects])

    # Calculate FNR and FPS
    fnr = false_negatives / len(true_objects) if len(true_objects) > 0 else 0  # FNR = FN / (TP + FN)
    fps = false_positives / len(detected_objects) if len(detected_objects) > 0 else 0  # FPS = FP / total detected objects

    return fnr, fps

def input_ground_truth(img_name):
    """Prompt user for ground truth input."""
    print(f"Please input the objects in the image '{img_name}' (comma-separated):")
    print("Available objects: person, car, bicycle, motorbike, bus, truck, dog, cat")
    input_data = input("Objects: ")
    objects = [obj.strip() for obj in input_data.split(',')]
    ground_truth[img_name] = objects

# Reinforcement Learning Model for Ethical Decision-Making
class EthicalDecisionModel(tf.keras.Model):
    def __init__(self):
        super(EthicalDecisionModel, self).__init__()
        # Define layers for RL model
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')  # Output: Ethical score adjustment

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Reinforcement Learning Agent
class EthicalRLAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.model = EthicalDecisionModel()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model(state)
        return np.argmax(act_values[0])  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model(next_state)[0])
            target_f = self.model(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

def integrate_rl_in_decision_process(detected_objects, ethical_scores):
    # Ensure ethical_scores is a dictionary
    if isinstance(ethical_scores, dict):
        state = np.array([list(ethical_scores.values())])  # Convert dictionary values to a list and wrap in an array
    else:
        # If ethical_scores is a float, we need to create a dictionary for it
        state = np.array([[ethical_scores]])  # Wrap it into a list as a 1x1 array

    agent = EthicalRLAgent(action_size=2)  # Assume two actions: Adjust or maintain ethical priority
    action = agent.act(state)  # Decide whether to adjust ethical priorities
    
    if action == 0:  # Adjust priorities
        adjusted_ethical_scores = adjust_ethical_priorities(detected_objects, ethical_scores)
        return adjusted_ethical_scores
    else:
        return ethical_scores  # Keep the current ethical priorities


def adjust_ethical_priorities(detected_objects, ethical_scores):
    # Ensure ethical_scores is a dictionary
    if isinstance(ethical_scores, dict):
        adjusted_scores = ethical_scores.copy()  # Copy the dictionary if it is a dict
    else:
        # If ethical_scores is a float, create a dictionary for it
        adjusted_scores = {'person': ethical_scores, 'car': 0.8, 'bicycle': 0.9}  # Default values for other objects

    # Example logic: give higher priority to pedestrian detection
    if 'person' in detected_objects:
        adjusted_scores['person'] += 0.2

    return adjusted_scores



# IoT Integration: Assume real-time GPS data for ethical decision making
def integrate_gps_data_to_adjust_priorities(gps_data, ethical_scores):
    # Example: Adjust ethical decision-making based on GPS data (e.g., in pedestrian zones)
    if gps_data['zone'] == 'pedestrian':
        ethical_scores['person'] += 0.5  # Prioritize person detection in pedestrian zones
    return ethical_scores

# Example usage of RL and GPS Integration
ethical_scores = {'person': 1.2, 'car': 0.8, 'bicycle': 0.9}
gps_data = {'zone': 'pedestrian'}
detected_objects = ['person', 'car']

ethical_scores = integrate_gps_data_to_adjust_priorities(gps_data, ethical_scores)
adjusted_scores = ethical_scores
print("Adjusted Ethical Scores:", adjusted_scores)

def process_images_with_ethical_factor(image_files, gps_data=None):
    """Process all images, detect objects, and calculate ethical score along with bias score."""
    bias_scores = []
    ethical_scores = []
    differences = []
    fnrs = []
    fpss = []
    
    results = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)

        # Input ground truth from user
        input_ground_truth(img_name)

        img, blob, height, width = preprocess_image(img_path)
        class_ids, confidences, boxes = detect_objects_in_image(img, blob, height, width)
        
        # Calculate Bias score
        bias_score, detected_objects = calculate_bias_score(class_ids)
        bias_scores.append(bias_score)
        
        # Calculate Ethical Score
        ethical_score, ethical_detected_objects = calculate_ethical_score(class_ids)
        ethical_scores.append(ethical_score)

        # Integrate RL and GPS for ethical score adjustment
        ethical_score = integrate_rl_in_decision_process(detected_objects, ethical_score)

        # Calculate FNR and FPS
        true_objects = ground_truth.get(img_name, [])
        fnr, fps = calculate_fnr_fps(true_objects, detected_objects)

        # Convert detected objects list to a dictionary (e.g., {"person": 3, "dog": 1})
        detected_object_count = dict(Counter(detected_objects))

        # Compare ground truth and detected objects using Counter (to account for the count)
        true_object_count = dict(Counter(true_objects))

        # Calculate difference: missing objects and extra detected objects
        missing_objects = {obj: true_object_count[obj] - detected_object_count.get(obj, 0) for obj in true_object_count if true_object_count[obj] > detected_object_count.get(obj, 0)}
        extra_objects = {obj: detected_object_count[obj] - true_object_count.get(obj, 0) for obj in detected_object_count if detected_object_count[obj] > true_object_count.get(obj, 0)}

        # Store the differences
        differences.append(f"Missing: {missing_objects}, Extra: {extra_objects}")

        # Store FNR and FPS
        fnrs.append(fnr)
        fpss.append(fps)

        # Store the results with detected objects in dictionary format
        gps_info = gps_data.get(img_name, "No GPS Data") if gps_data else "No GPS Data"
        results.append([img_name, str(detected_object_count), bias_score, ethical_score, fnr, fps, f"Missing: {missing_objects}, Extra: {extra_objects}", gps_info])

    # Create DataFrame for tabular output
    df = pd.DataFrame(results, columns=["Image", "Detected Objects (Dictionary)", "Bias Score", "Ethical Score", "FNR", "FPS", "Difference with Ground Truth", "GPS Data"])
    print(df)
    # import streamlit as st
    # st.write(df)

    # # If using ace_tools (for displaying nicely)
    # tools.display_dataframe_to_user(name="Processed Image Results", dataframe=df)
    
    # Bias and Ethical Score Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(bias_scores)), bias_scores, label="Bias Score", color="blue")
    plt.plot(range(len(ethical_scores)), ethical_scores, label="Ethical Score", color="green")
    plt.xlabel('Image Index')
    plt.ylabel('Score')
    plt.title('Bias and Ethical Scores Across Images')
    plt.legend()
    plt.show()

# Example GPS data (this could be dynamic in a real-world scenario)
gps_data = {
    "image1.jpg": "Latitude: 37.7749, Longitude: -122.4194",
    "image2.jpg": "Latitude: 34.0522, Longitude: -118.2437",
    "image3.jpg": "Latitude: 40.7128, Longitude: -74.0060"
}

# Run object detection, calculate bias scores, ethical scores, FNR, FPS for each image
process_images_with_ethical_factor(image_files, gps_data=gps_data)
