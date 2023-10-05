# Import necessary libraries
import cv2
from flask import Flask, render_template, Response

# Create a Flask web application
app = Flask(__name__)

# Load class names from a file (coco.names)
classNames = []
classFile = "/home/pi1/Desktop/webStreamingCV/coco.names"  # Update with your path to the coco.names file

# Read and store class names from the file
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Define paths for the configuration and weights of the pre-trained model
configPath = "/home/pi1/Desktop/webStreamingCV/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Update with your path
weightsPath = "/home/pi1/Desktop/webStreamingCV/frozen_inference_graph.pb"  # Update with your path

# Create a deep learning model for object detection using OpenCV
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to detect objects in an image
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    # Draw bounding boxes and labels on the image
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

# Function to generate video frames with object detection
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Call the object detection function on each frame
        # Remove objects=['whatever'] for the code to scan for everything
        frame, _ = getObjects(frame, 0.45, 0.2, draw=True, objects=['bottle'])
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define the index route for rendering a template
@app.route('/')
def index():
    return render_template('index.html')

# Define the video_feed route for streaming video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask application if this script is run directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

