# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
from itertools import count
import dlib
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}
x_values = []
Angry = []
Disgust = []
Fear = []
Happy = []
Sad = []
Suprise = []
Neutral = []
index = count() 


def animate(list_data):
    x_values.append(next(index))    
    Angry.append(list_data[0])
    Disgust.append(list_data[1])
    Fear.append(list_data[2])
    Happy.append(list_data[3])
    Sad.append(list_data[4])
    Suprise.append(list_data[5])
    Neutral.append(list_data[6])
    
    # plt.cla()
    plt.plot(x_values, Angry, color="#2a46c1", label="Angry")
    plt.plot(x_values, Disgust, color="#31afa4", label="Disgust")
    plt.plot(x_values, Fear, color="#9b3428", label="Fear")
    plt.plot(x_values, Happy, color="#1ca417", label="Happy")
    plt.plot(x_values, Sad, color="#175da4", label="Sad")
    plt.plot(x_values, Suprise, color="#61e5da", label="Suprise")
    plt.plot(x_values, Neutral, color="#c8486c", label="Neutral")
    plt.pause(0.05)
    
def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = 'models/emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

cap = cv2.VideoCapture(0)

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22,
                                 (capWidth, capHeight))
plt.axis([0, 400, 0, 1])
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))

    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        print(emotion_prediction)
        emotion_probability = np.max(emotion_prediction)
        # if (emotion_probability > 0.36):
        emotion_label_arg = np.argmax(emotion_prediction)
        color = emotions[emotion_label_arg]['color']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                    color,
                    thickness=2)
        cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40),
                        color, -1)
        cv2.putText(frame, emotions[emotion_label_arg]['emotion'],
                    (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        
        x = str(round(emotion_prediction[0][0], 2))
        cv2.putText(frame, f"Angry: {x}" ,
                    ( 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[0]['color'], 1, cv2.LINE_AA)
        x = str(round(emotion_prediction[0][1], 2))
        cv2.putText(frame, f"Disgust: {x}" ,
                    ( 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[1]['color'], 1, cv2.LINE_AA)
        x = str(round(emotion_prediction[0][2], 2))
        cv2.putText(frame, f"Fear: {x}" ,
                    ( 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[2]['color'], 1, cv2.LINE_AA)
        x = str(round(emotion_prediction[0][3], 2))
        cv2.putText(frame, f"Happy: {x}" ,
                    ( 20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[3]['color'], 1, cv2.LINE_AA)
        x = str(round(emotion_prediction[0][4], 2))
        cv2.putText(frame, f"Sad: {x}" ,
                    ( 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[4]['color'], 1, cv2.LINE_AA)
        x = str(round(emotion_prediction[0][5], 2))
        cv2.putText(frame, f"Suprise: {x}" ,
                    ( 20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[5]['color'], 1, cv2.LINE_AA)
        x = str(round(emotion_prediction[0][6], 2))
        cv2.putText(frame, f"Neutral: {x}" ,
                    ( 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    emotions[6]['color'], 1, cv2.LINE_AA)
        # ani = FuncAnimation(plt.gcf(), animate(emotion_prediction[0]), 10)
        animate(emotion_prediction[0])
        # else:
        #     color = (255, 255, 255)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # plt.tight_layout()

    # plt.pause(0.1)
    # plt.close()
    if args["isVideoWriter"] == True:
        videoWrite.write(frame)
            



    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
plt.show(block=False)
cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
