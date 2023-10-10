import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
user_count = 0

classNames = []
classFile = "/home/pi1/Desktop/webStreamingCV/coco.names"  # Update with your path to the coco.names file

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi1/Desktop/webStreamingCV/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Update with your path
weightsPath = "/home/pi1/Desktop/webStreamingCV/frozen_inference_graph.pb"  # Update with your path

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


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
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame, _ = getObjects(frame, 0.45, 0.2, draw=True, objects=['bottle'])
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    global user_count
    user_count += 1
    emit('update_count', {'count': user_count}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    global user_count
    user_count -= 1
    emit('update_count', {'count': user_count}, broadcast=True)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)
