
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict
from face import recognize_faces, load_database  # Import dari file lain

def evaluate_model(database_path, test_dataset_path):
    database = load_database(database_path)
    y_true = []
    y_pred = []
    for person_name in os.listdir(test_dataset_path):
        person_dir = os.path.join(test_dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            predictions, similarities, face_locations = recognize_faces(img, database)
            if len(predictions) == 1:
                y_true.append(person_name)
                y_pred.append(predictions[0])
            else:
                print(f"Warning: {img_path} - Detected {len(predictions)} faces (expected 1), skipped.")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}")
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true + y_pred))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true + y_pred), yticklabels=np.unique(y_true + y_pred))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
def evaluate_metrics_from_cm(cm):
    """
    Hitung TP, FP, FN, TN untuk multi-class classification dari confusion matrix.
    Mengembalikan dictionary per class.
    """
    metrics = {}
    num_classes = cm.shape[0]
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        metrics[i] = {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Precision': TP / (TP + FP) if TP + FP > 0 else 0,
            'Recall': TP / (TP + FN) if TP + FN > 0 else 0,
            'F1-Score': 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0,
            'Accuracy': (TP + TN) / cm.sum() if cm.sum() > 0 else 0
        }
    return metrics
    
def evaluate_from_webcam_per_person(database_path, people_list, samples_per_person=5):
    database = load_database(database_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    y_true = []
    y_pred = []
    per_person_results = defaultdict(lambda: {'true': [], 'pred': []})

    for person in people_list:
        print(f"\n[INFO] Uji untuk: {person}")
        print(f"Ambil {samples_per_person} gambar dari webcam. Tekan 'SPACE' untuk capture.")

        captured = 0
        while captured < samples_per_person:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            predictions, similarities, face_locations = recognize_faces(frame, database)
            frame_disp = frame.copy()
            for name, sim, loc in zip(predictions, similarities, face_locations):
                start_point, end_point = loc
                label = f"{name}: {sim:.2f}"
                color = (0, 255, 0) if name == person else (0, 0, 255)
                cv2.rectangle(frame_disp, start_point, end_point, color, 2)
                cv2.putText(frame_disp, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Webcam Test", frame_disp)

            key = cv2.waitKey(1)
            if key == 32:  # Space
                if len(predictions) == 1:
                    y_true.append(person)
                    y_pred.append(predictions[0])
                    per_person_results[person]['true'].append(person)
                    per_person_results[person]['pred'].append(predictions[0])
                    captured += 1
                    print(f"[CAPTURED] Gambar {captured}/{samples_per_person}: Prediksi = {predictions[0]}")
                else:
                    print("[SKIP] Wajah tidak terdeteksi atau lebih dari satu.")
            elif key == 27:  # ESC
                print("Ujian dibatalkan.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    # === GLOBAL EVALUATION ===
    print("\n=== GLOBAL CONFUSION MATRIX & REPORT ===\n")
    labels = sorted(list(set(y_true + y_pred)))
    print(classification_report(y_true, y_pred, labels=labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Webcam-Based Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # === Tambahkan Evaluasi TP, FP, FN, TN, Precision, Recall, F1, Accuracy ===
    print("\n=== DETAILED METRICS PER CLASS ===\n")
    metrics = evaluate_metrics_from_cm(cm)
    for i, label in enumerate(labels):
        m = metrics[i]
        print(f"Class: {label}")
        print(f"  TP: {m['TP']}, FP: {m['FP']}, FN: {m['FN']}, TN: {m['TN']}")
        print(f"  Precision: {m['Precision']:.2f}")
        print(f"  Recall:    {m['Recall']:.2f}")
        print(f"  F1-Score:  {m['F1-Score']:.2f}")
        print(f"  Accuracy:  {m['Accuracy']:.2f}")
        print()

    # === PER PERSON REPORT ===
    print("\n=== PER PERSON PERFORMANCE ===\n")
    for person, data in per_person_results.items():
        true = data['true']
        pred = data['pred']
        acc = accuracy_score(true, pred)
        errors = [(t, p) for t, p in zip(true, pred) if t != p]

        print(f"--- {person} ---")
        print(f"Jumlah Sampel: {len(true)}")
        print(f"Benar: {true.count(person)}")
        print(f"Salah: {len(errors)}")
        print(f"Akurasi: {acc:.2%}")
        if errors:
            print("Prediksi Salah:")
            for e in errors:
                print(f"  - Asli: {e[0]} → Prediksi: {e[1]}")
        print()
        
if __name__ == "__main__":
    test_dataset_path = "test_dataset"
    database_path = "face_recognition_database.pkl"


    # Option : Evaluasi model dengan Confusion Matrix
    # evaluate_model(database_path, test_dataset_path)
    
    # Jalankan face recognition dan evaluasi langsung dari webcam
    people_to_test = ["andra", "tian",]  # Ganti sesuai data orang di database
    evaluate_from_webcam_per_person("face_recognition_database.pkl", people_to_test, samples_per_person=5)