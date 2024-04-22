import cv2
import os

# Impostare il percorso della cartella contenente i video
video_path = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video/Move Forward"

# Ottenere la lista di tutti i file nella cartella
file_list = os.listdir(video_path)

# Creare una lista di tutti i video nella cartella
video_list = [f for f in file_list if f.endswith('.mp4') or f.endswith('.avi')]

# Ottenere il numero totale di video nella cartella
num_videos = len(video_list)

# Impostare la scala di zoom desiderata (0.5 = zoom al 50%)
zoom_scale = 0.9

# Ciclare attraverso tutti i video nella cartella
for i, video in enumerate(video_list):
    # Aprire il video
    cap = cv2.VideoCapture(os.path.join(video_path, video))
    

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # ottenere le dimensioni dei frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # creare il file video di output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Creare un oggetto VideoWriter per il video flippato
    out_path = os.path.join(video_path, f"{num_videos+i+1}.avi")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while(cap.isOpened()):
        # catturare un singolo frame
        ret, frame = cap.read()
        if ret == True:
            # applicare lo zoom
            new_frame = cv2.resize(frame, (0, 0), fx=1.1, fy=1.1)

            # ritagliare l'immagine
            crop = new_frame[0:height, 0:width]

            # scrivere il nuovo frame nel video
            out.write(crop)

        else:
            break

    # chiudere tutto
    cap.release()
    out.release()
print(f"Sono stati aggiunti {i+1} video zoomati alla cartella.")
