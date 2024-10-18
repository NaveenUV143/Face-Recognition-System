import os
import cv2
import numpy as np

def collect_training_data(data_dir):
    face_images = []
    labels = []
    
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            label_dict[current_label] = person_name
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_images.append(gray_image)
                labels.append(current_label)
            current_label += 1

    return face_images, np.array(labels), label_dict

def train_model():
    data_dir = "data"
    face_images, labels, label_dict = collect_training_data(data_dir)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_images, labels)

    face_recognizer.save("face_recognizer.yml")
    np.save("label_dict.npy", label_dict)

    print("Training completed!")

if __name__ == "__main__":
    train_model()
