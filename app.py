import cv2
import numpy as np

def load_model():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("face_recognizer.yml")
    label_dict = np.load("label_dict.npy", allow_pickle=True).item()
    return face_recognizer, label_dict

def recognize_faces():
    face_recognizer, label_dict = load_model()
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_id, confidence = face_recognizer.predict(gray_frame[y:y+h, x:x+w])
            
            if confidence < 100:
                name = label_dict[face_id]
                confidence_text = f"{name} - {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"{round(100 - confidence)}%"

            cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
