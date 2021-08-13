import cv2
from PIL import Image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

p = transforms.Compose([transforms.Resize((96, 96)),
                        transforms.ToTensor(),
                        ])
# from imageai import Detection

app = Flask(__name__)

# modelpath = "yolo.h5"
# yolo = Detection.ObjectDetection()
# yolo.setModelTypeAsYOLOv3()
# yolo.setModelPath(modelpath)
# yolo.loadModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speech_detector = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=(3, 3)),
    nn.Conv2d(16, 64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(64, 128, kernel_size=(3, 3)),
    nn.Conv2d(128, 256, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.LazyLinear(512),
    nn.LazyLinear(1)
).to(device)
speech_detector.load_state_dict(torch.load("model2850.pt"))
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


def predict_image(image):
    image_tensor = p(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = speech_detector(input)
    index = output.data.cpu().numpy() >= 0.7
    return index

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # img, preds = yolo.detectCustomObjectsFromImage(input_image=frame,
            #           custom_objects=None, input_type="array",
            #           output_type="array",
            #           minimum_percentage_probability=70,
            #           display_percentage_probability=False,
            #           display_object_name=True)
            # print(frame.shape)
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            image = ""
            print(faces)
            if len(faces) == 0:
                image = frame
            else:
                (x, y, w, h) = faces[0]
                if predict_image(Image.fromarray(frame[y:y+h,x:x+w])):
                    image = cv2.rectangle(
                        frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    image = frame
                print(predict_image(Image.fromarray(frame[y:y+h,x:x+w])))

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    return render_template('Write-your-story-with-AI.html', generated=None)

@app.route('/')
def home():
    return render_template('writer_home.html', generated=None)


if __name__ == '__main__':
    app.run(debug=True)
