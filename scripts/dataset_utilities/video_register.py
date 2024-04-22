import os
import cv2
import time, rospy

# Countdown Function
def start_countdown(num_of_secs):

    print("\nAcquisition Starts in:")

    # Wait Until 0 Seconds Remaining
    while (not rospy.is_shutdown() and num_of_secs != 0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        rospy.sleep(1)
        num_of_secs -= 1

    print("\nSTART\n")

def rest_countdown(num_of_secs):


    # Wait Until 0 Seconds Remaining
    while (not rospy.is_shutdown() and num_of_secs != 0):
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format)
        rospy.sleep(1)
        num_of_secs -= 1


# Input del numero di video da registrare
num_videos = int(input("Inserisci il numero di video che vuoi registrare: "))

#Inserisci il tipo di video che vuoi registrare

# gesture_type_to_add = "Stop"
# gesture_type_to_add = "No Gesture"
# gesture_type_to_add = "Point at"
# gesture_type_to_add = "Thumb Up"
# gesture_type_to_add = "Move Forward"
# gesture_type_to_add = "Move Right"
# gesture_type_to_add = "Move Left"
# gesture_type_to_add = "Move Backward"
gesture_type_to_add = "Resume"
# gesture_type_to_add = "Pause"

'''
Importante: Se sei in un ambiente con un illuminazione di merda si abbassa il framerate della videocamera e quando processi i video hai le shape sminchiate
  v4l2-ctl --list-devices

'''

#Impostazioni della registrazione video


video_duration = 3  
pause = 3    #Pausa tra i video in secondi
video_path = "/home/alberto/catkin_ws/src/mediapipe_gesture_recognition/dataset/Video"  
gesture_video_path = os.path.join(video_path, gesture_type_to_add)
video_format = ".avi"
# Controllo se la cartella esiste
if not os.path.exists(gesture_video_path):
    os.makedirs(gesture_video_path)
# Ottenimento del numero del video più alto nella cartella
existing_videos = [f for f in os.listdir(gesture_video_path) if f.endswith(video_format)]
if existing_videos:
    latest_video = max([int(f.split(".")[0]) for f in existing_videos])
else:
    latest_video = 0
print("I'm recording the gesture '{}' ".format(gesture_type_to_add))
Start_countdown(3)


for i in range(num_videos):

    webcam = 6
    # Preparazione della cattura video
    video_name = str(latest_video + i + 1) + video_format
    video_path = os.path.join(gesture_video_path, video_name)
    capture = cv2.VideoCapture(webcam)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))
    print("Video number | {}".format(i+ 1))
    # Registrazione del video
    start_time = time.time()
    frame_count = 0
    while int(time.time() - start_time) < int(video_duration):
        ret, frame = capture.read()
        cv2.imshow(gesture_type_to_add, cv2.flip(frame, 1))
        cv2.waitKey(1)
        if ret:
            out.write(frame)
        else:
            break
        frame_count += 1
    capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video Frames: {} and video duration {}".format(frame_count, video_duration))
    # Pausa tra i video
    time.sleep(pause)

print("End videos")


