#!/usr/bin/env python
# coding: utf-8

# # OPENCV PROJECT
# ## Hand Detector and Control Bar
# 
# #### PRESS ESC TO EXIT AND CLOSE THE WINDOW

# In[1]:


get_ipython().system('pip install mediapipe')


# In[ ]:


## Import the necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from IPython.display import display, clear_output

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Initialize hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Get landmarks for the first hand (assuming only one hand is in the frame)
            landmarks = results.multi_hand_landmarks[0].landmark

            # Extract specific landmarks for control (e.g., thumb tip and index finger tip)
            thumb_tip = (int(landmarks[4].x * frame.shape[1]), int(landmarks[4].y * frame.shape[0]))
            index_tip = (int(landmarks[8].x * frame.shape[1]), int(landmarks[8].y * frame.shape[0]))

            # Calculate the distance between thumb tip and index finger tip
            distance = calculate_distance(thumb_tip, index_tip)

            # Display the distance
            cv2.putText(frame, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Detector Control Bar', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




