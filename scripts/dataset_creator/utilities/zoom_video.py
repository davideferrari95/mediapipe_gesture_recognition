import cv2, os

# leggere il video

video_path = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video/Thumb Up"


# Controllo se la cartella esiste
if not os.path.exists(video_path):
    os.makedirs(video_path)

# Ottenimento del numero del video più alto nella cartella
existing_videos = [f for f in os.listdir(video_path) if f.endswith(video_path)]

if existing_videos:
    latest_video = max([int(f.split(".")[0]) for f in existing_videos])
else:
    latest_video = 0


for video in video_path:

    cap = cv2.VideoCapture(video)

    # ottenere le dimensioni dei frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # creare il file video di output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(latest_video, fourcc, 20.0, (width, height))

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
    cv2.destroyAllWindows()
