import os, cv2, time, rospy
from termcolor import colored

# Package Path
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

class VideoRecorder:

    """ Video Recorder Class - Dataset Videos """

    """ Important! If you record in a low-light environment, the camera's frame rate drops, and you will have messed up shapes. """

    """ Get Webcam Number: v4l2-ctl --list-devices """

    # Registration Settings
    video_duration, pause, video_format = 3, 3, '.avi'
    video_path = os.path.join(PACKAGE_PATH, 'dataset/Video')
    webcam_number = 0

    def __init__(self):

        # Number of Videos to Record
        num_videos = int(input('\nInsert the number of videos you want to record: '))

        # Available Gestures List
        gestures = ['Stop', 'No Gesture', 'Point at', 'Thumb Up', 'Move Forward', 'Move Right', 'Move Left', 'Move Backward', 'Resume', 'Pause']

        # Print Available Gestures
        print(colored('\nAvailable Gestures:\n', 'green'))
        for i, gesture in enumerate(gestures, start=1):
            print(f'{i}. {gesture}')

        # Ask for the Gesture to Record
        while True:

            try:

                # Selected Gesture Index
                selected_index = int(input('\nInsert the number of the gesture to record: '))

                # Check if the Selected Index is Valid
                if 1 <= selected_index <= len(gestures): selected_gesture = gestures[selected_index - 1]; break
                else: print('Insert a valid number corresponding to a gesture in the list.')

            except ValueError: print('Insert a valid number.')

        # Check if the Gesture Folder Exists
        if not os.path.exists(os.path.join(self.video_path, selected_gesture)): os.makedirs(os.path.join(self.video_path, selected_gesture))

        # Get the Highest Video Number in the Folder
        existing_videos = [f for f in os.listdir(os.path.join(self.video_path, selected_gesture)) if f.endswith(self.video_format)]
        latest_video = max([int(f.split(".")[0]) for f in existing_videos]) if existing_videos else 0

        # Record the Videos
        self.record(selected_gesture, latest_video, num_videos)

    def countdown(self, num_of_secs:int):

        """ Countdown Function """

        print("\nAcquisition Starts in:")

        # Wait Until 0 Seconds Remaining
        while (not rospy.is_shutdown() and num_of_secs != 0):

            # Print the Remaining Time
            print('{:02d}:{:02d}'.format(*divmod(num_of_secs, 60)))

            # Wait 1 Second
            rospy.sleep(1)
            num_of_secs -= 1

        print("\nSTART\n")

    def record(self, gesture:str, latest_video:int, num_videos:int):

        """ Record Videos """

        print(f'Recording Gesture: "{gesture}"')
        self.countdown(3)

        # Record the Videos
        for i in range(num_videos):

            # Prepare Video Path - Video Number + Video Format
            video_path = os.path.join(os.path.join(self.video_path, gesture), str(latest_video + i + 1) + self.video_format)

            # Prepare Video Capture
            capture, fps = cv2.VideoCapture(self.webcam_number), 30
            frame_width, frame_height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))
            print(f'Video Number | {i+1}')

            # Record the Video
            start_time, frame_count = time.time(), 0

            while int(time.time() - start_time) < int(self.video_duration):

                # Read the Frame
                ret, frame = capture.read()

                # Show the Frame and Wait
                cv2.imshow(gesture, cv2.flip(frame, 1))
                cv2.waitKey(1)

                # Write the Frame
                if ret: out.write(frame); frame_count += 1
                else: break

            # Release Video Capture and Writer
            capture.release(); out.release()
            cv2.destroyAllWindows()

            print(f'Video Frames: {frame_count} and Video Duration {self.video_duration}')

            # Pause Between Videos
            time.sleep(self.pause)

        # End Recording
        print(f'End Recording Videos for Gesture: "{gesture}"')

if __name__ == '__main__':

    # Initialize the Video Recorder
    VideoRecorder()
