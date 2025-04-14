import cv2
import numpy as np
import mtcnn
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import time
from scipy.spatial.distance import cosine

# Initialize MTCNN for face detection
face_detector = mtcnn.MTCNN()

# Load the FaceNet model
facenet_model = load_model('facenet_keras.h5')

# Initialize L2 normalizer
l2_normalizer = Normalizer('l2')

# Configuration parameters
detection_confidence_threshold = 0.95
required_size = (160, 160)  # FaceNet requires 160x160 input
similarity_threshold = 0.5  # Cosine similarity threshold (lower means more similar)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def detect_face(img):
    faces = face_detector.detect_faces(img)
    face_boxes = []
    face_scores = []
    for face in faces:
        confidence = face['confidence']
        if confidence >= detection_confidence_threshold:
            x, y, w, h = face['box']
            face_boxes.append([x, y, w, h])
            face_scores.append(confidence)
    return face_boxes, face_scores

def normalize_faces(img, face_boxes):
    face_imgs = []
    face_locations = []
    for box in face_boxes:
        face_img, start_point, end_point = get_face(img, box)
        if face_img.size != 0:
            face_img = cv2.resize(face_img, required_size)
            face_imgs.append(face_img)
            face_locations.append((start_point, end_point))
    return face_imgs, face_locations

def get_embeddings(face_imgs):
    face_imgs = [cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) for face_img in face_imgs]
    face_imgs = np.array([img.astype('float32') for img in face_imgs])
    face_imgs = (face_imgs - 127.5) / 128.0
    embeddings = facenet_model.predict(face_imgs)
    embeddings = l2_normalizer.transform(embeddings)
    return embeddings

def create_dataset(dataset_path):
    embeddings = []
    names = []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                face_boxes, _ = detect_face(img)
                if len(face_boxes) == 1:
                    face_imgs, _ = normalize_faces(img, face_boxes)
                    if len(face_imgs) == 1:
                        face_embedding = get_embeddings(face_imgs)[0]
                        embeddings.append(face_embedding)
                        names.append(person_name)
                else:
                    print(f"Warning: Found {len(face_boxes)} faces in {img_path}, expected 1. Skipping image.")
    database = {'embeddings': np.array(embeddings), 'names': np.array(names)}
    return database

def save_database(database, database_path):
    with open(database_path, 'wb') as file:
        pickle.dump(database, file)

def load_database(database_path):
    with open(database_path, 'rb') as file:
        database = pickle.load(file)
    return database

def recognize_face_cosine_similarity(embedding, database, threshold=similarity_threshold):
    if len(database['embeddings']) == 0:
        return "Unknown", 0.0
    similarities = []
    for ref_embedding in database['embeddings']:
        similarity = 1 - cosine(embedding, ref_embedding)
        similarities.append(similarity)
    idx = np.argmax(similarities)
    max_similarity = similarities[idx]
    if max_similarity >= threshold:
        return database['names'][idx], max_similarity
    else:
        return "Unknown", max_similarity

def recognize_faces(img, database):
    face_boxes, face_scores = detect_face(img)
    if len(face_boxes) == 0:
        return [], [], []
    face_imgs, face_locations = normalize_faces(img, face_boxes)
    if len(face_imgs) == 0:
        return [], [], []
    embeddings = get_embeddings(face_imgs)
    predictions = []
    similarities = []
    for embedding in embeddings:
        name, similarity = recognize_face_cosine_similarity(embedding, database)
        predictions.append(name)
        similarities.append(similarity)
    return predictions, similarities, face_locations

def add_person_to_dataset(dataset_path, person_name, capture_count=10):
    person_dir = os.path.join(dataset_path, person_name)
    os.makedirs(person_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    image_count = 0
    last_capture_time = time.time() - 2
    print(f"Adding {person_name} to dataset. Capturing {capture_count} images...")
    while image_count < capture_count:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        cv2.imshow("Capture", frame)
        face_boxes, _ = detect_face(frame)
        for box in face_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        current_time = time.time()
        if len(face_boxes) == 1 and current_time - last_capture_time >= 2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = os.path.join(person_dir, f"{timestamp}.jpg")
            cv2.imwrite(img_path, frame)
            image_count += 1
            last_capture_time = current_time
            print(f"Captured image {image_count}/{capture_count}")
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

def run_recognition(database_path, similarity_threshold=0.7):
    database = load_database(database_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("Starting face recognition. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        predictions, similarities, face_locations = recognize_faces(frame, database)
        for name, similarity, loc in zip(predictions, similarities, face_locations):
            start_point, end_point = loc
            color = (0, 255, 0) if similarity >= similarity_threshold else (0, 0, 255)
            label = f"{name}: {similarity:.2f}" if similarity >= similarity_threshold else f"Unknown: {similarity:.2f}"
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_path = "face_dataset"
    test_dataset_path = "test_dataset"
    database_path = "face_recognition_database.pkl"

    os.makedirs(dataset_path, exist_ok=True)

    # Option 1: Tambahkan orang baru ke dataset
    # add_person_to_dataset(dataset_path, "nama",capture_count=10)

    # Option 2: Buat dan simpan database dari dataset pelatihan
    database = create_dataset(dataset_path)
    save_database(database, database_path)

    # Option 3: Jalankan face recognition secara real-time
    run_recognition(database_path, similarity_threshold=0.7)

    # Option 4: Evaluasi model dengan Confusion Matrix
    # evaluate_model(database_path, test_dataset_path)
    
    # Jalankan face recognition dan evaluasi langsung dari webcam
    # people_to_test = ["andra", "tian",]  # Ganti sesuai data orang di database
    # evaluate_from_webcam_per_person("face_recognition_database.pkl", people_to_test, samples_per_person=5)
