from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread, Event
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
import numpy as np
import imutils
import time
import dlib
import cv2
import pyttsx3
import logging
import os
import base64

# Suppress MSMF warnings
os.environ["OPENCV_LOG_LEVEL"]="ERROR"
logging.getLogger("opencv").setLevel(logging.ERROR)

app = Flask(__name__)

# Global variables
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0
TOTAL = 0
ALARM_ON = False
YAWN_COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False
video_stream = None
stream_active = False

# Initialize your detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/ubuntu/dddd/shape_predictor_68_face_landmarks.dat")

# Add these helper functions from your drowsy.py
def compute_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def check_drowsy(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = compute_ear(leftEye)
    rightEAR = compute_ear(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def check_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def alarm(msg):
    global alarm_status, alarm_status2, saying
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(msg)
    engine.runAndWait()

def initialize_camera():
    """Initialize camera with working configuration"""
    global video_stream
    
    try:
        # Use VideoStream with a warm-up period - this has proven most reliable
        video_stream = VideoStream(src=0).start()
        time.sleep(2.0)  # Generous warm-up time
        
        # Test frame grab
        frame = video_stream.read()
        if frame is not None:
            return True
            
        video_stream.stop()
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        if video_stream is not None:
            video_stream.stop()
    
    return False

def generate_frames():
    global video_stream, COUNTER, stream_active, alarm_status, alarm_status2
    
    try:
        if video_stream is None:
            if not initialize_camera():
                raise Exception("Failed to initialize camera")
            stream_active = True

        while stream_active:
            frame = video_stream.read()
            if frame is None:
                continue
                
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            rects = detector(gray, 0)

            # Process each detected face
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Calculate eye aspect ratio
                eye = check_drowsy(shape)
                ear = eye[0]
                leftEye = eye[1]
                rightEye = eye[2]

                # Calculate lip distance
                distance = check_yawn(shape)

                # Draw eye contours
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # Draw lip contour
                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                # Check for drowsiness
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_status:
                            alarm_status = True
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_status = False

                # Check for yawning
                if distance > YAWN_THRESH:
                    if not alarm_status2:
                        alarm_status2 = True
                    cv2.putText(frame, "Yawn Alert", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    alarm_status2 = False

                # Display metrics
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Encode and yield frame with alert headers
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'X-Drowsy: ' + str(alarm_status).encode() + b'\r\n'
                   b'X-Yawn: ' + str(alarm_status2).encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        if video_stream is not None:
            video_stream.stop()
            video_stream = None
        stream_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    global stream_active, video_stream
    
    if video_stream is not None:
        video_stream.stop()
        video_stream = None
    
    stream_active = True
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream')
def stop_stream():
    global video_stream, stream_active, COUNTER, alarm_status, alarm_status2
    
    stream_active = False
    COUNTER = 0
    alarm_status = False
    alarm_status2 = False
    
    if video_stream is not None:
        video_stream.stop()
        video_stream = None
    
    return jsonify({"status": "success"})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global COUNTER, TOTAL, ALARM_ON, YAWN_COUNTER
    
    try:
        # Get frame from client
        frame_data = request.json['frame']
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        drowsy_alert = False
        yawn_alert = False
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Check for drowsiness
            ear, leftEye, rightEye = check_drowsy(shape)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            if ear < 0.25:
                COUNTER += 1
                if COUNTER >= 10:
                    if not ALARM_ON:
                        ALARM_ON = True
                        TOTAL += 1
                    drowsy_alert = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
            
            # Check for yawning
            distance = check_yawn(shape)
            
            # Draw mouth contours
            mouth_hull = cv2.convexHull(shape[48:60])
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
            
            if distance > 25:
                YAWN_COUNTER += 1
                yawn_alert = True
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                YAWN_COUNTER = 0
                
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_frame': f'data:image/jpeg;base64,{processed_frame}',
            'drowsy_alert': drowsy_alert,
            'yawn_alert': yawn_alert,
            'ear_value': float(ear) if 'ear' in locals() else 0.0,
            'yawn_value': float(distance) if 'distance' in locals() else 0.0
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

# Add a route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 