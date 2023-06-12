import cv2
import os

# Impostare il percorso della cartella contenente i video
video_path = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video/Resume"

# Ottenere la lista di tutti i file nella cartella
file_list = os.listdir(video_path)

# Creare una lista di tutti i video nella cartella
video_list = [f for f in file_list if f.endswith('.mp4') or f.endswith('.avi')]

# Ottenere il numero totale di video nella cartella
num_videos = len(video_list)

# Ciclare attraverso tutti i video nella cartella
for i, video in enumerate(video_list):
    
    # Aprire il video
    cap = cv2.VideoCapture(os.path.join(video_path, video))
    
    # Ottenere le informazioni sul video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Creare un oggetto VideoWriter per il video flippato
    out_path = os.path.join(video_path, f"{num_videos+i+1}.avi")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), True)

    # Ciclare attraverso tutti i frame del video
    for j in range(num_frames):
        # Leggere il frame dal video
        ret, frame = cap.read()
        
        # Invertire orizzontalmente il frame
        flipped_frame = cv2.flip(frame, 1)

        # Scrivere il frame flippato nel video di output
        out.write(flipped_frame)

    # Chiudere l'oggetto VideoWriter
    out.release()

    # Chiudere il video
    cap.release()

print(f"Sono stati aggiunti {i+1} video flippati alla cartella.")
