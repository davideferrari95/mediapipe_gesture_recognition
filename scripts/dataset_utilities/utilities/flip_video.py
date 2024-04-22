import cv2

# Definisci il nome del file video da leggere e scrivere
input_file = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/useful_scripts/3.avi"
output_file = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/useful_scripts/3_flip.avi"

# Apri il file video in modalità lettura
cap = cv2.VideoCapture(input_file)

# Prendi le informazioni sul video sorgente (lunghezza, altezza, fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Creare un'istanza del writer video per scrivere il file video modificato
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Leggi il video frame per frame, specchia ogni frame e scrivi il nuovo frame nel file video di output
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    mirror_frame = cv2.flip(frame, 1)
    out.write(mirror_frame)

# Chiudi file video sorgente e file video di output
cap.release()
out.release()