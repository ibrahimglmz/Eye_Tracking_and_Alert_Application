import cv2
import mediapipe as mp

# Mediapipe ve OpenCV başlatma
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eşik değerler
EYE_CLOSED_THRESHOLD = 0.25
FACE_NOT_DETECTED_THRESHOLD = 0.5  # Ekrandaki yüz oranı eşik değeri
EYE_OPEN_DURATION_THRESHOLD = 10  # Gözlerin kapalı kalma süresi (kare sayısı)

# Kamera açma
cap = cv2.VideoCapture(0)
eye_closed_frame_count = 0
alert_displayed = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera verisi alinamadi.")
        break

    # BGR görüntüyü RGB'ye çevir
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Performansı arttırmak için bu adımı yapıyoruz
    image.flags.writeable = False

    # Yüz ağını işleme
    results = face_mesh.process(image)

    # Görüntüyü tekrar yazılabilir hale getiriyoruz
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_detected = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_detected = True

            # Sadece landmark'ları çiz
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            # Gözlerin kapanma durumunu tespit etme
            left_eye_landmarks = [face_landmarks.landmark[i] for i in range(33, 133)]  # Sol göz landmark'ları
            right_eye_landmarks = [face_landmarks.landmark[i] for i in range(362, 463)]  # Sağ göz landmark'ları

            left_eye_open_ratio = (left_eye_landmarks[5].y - left_eye_landmarks[1].y) / (
                        left_eye_landmarks[3].x - left_eye_landmarks[0].x)
            right_eye_open_ratio = (right_eye_landmarks[5].y - right_eye_landmarks[1].y) / (
                        right_eye_landmarks[3].x - right_eye_landmarks[0].x)

            average_eye_open_ratio = (left_eye_open_ratio + right_eye_open_ratio) / 2

            if average_eye_open_ratio < EYE_CLOSED_THRESHOLD:
                eye_closed_frame_count += 1
            else:
                eye_closed_frame_count = 0

            if eye_closed_frame_count > EYE_OPEN_DURATION_THRESHOLD:
                if not alert_displayed:
                    print("Gözler kapali, lütfen kameraya bakin!")
                    alert_displayed = True
                cv2.putText(image, 'UYARI: Gozler kapali!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
            else:
                alert_displayed = False

    if not face_detected:
        cv2.putText(image, 'UYARI: Kameraya bakmalisiniz!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    # Görüntüyü ekranda göster
    cv2.imshow('Goz ve Kamera Uyari Sistemi', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
