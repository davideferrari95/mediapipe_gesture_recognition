import os, cv2
from tqdm import tqdm
from natsort import natsorted

class FlipZoomDataset:

    def __init__(self):

        # Package Path
        FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))

        # Video Path Folder
        video_path = os.path.join(FOLDER, 'dataset/Gestures')

        for gesture in os.listdir(video_path):

            print(f'\nFlip and Zoom Videos for Gesture: {gesture}\n')

            # Flip and Zoom (Concatenate) Videos
            self.flip_videos(os.path.join(video_path, gesture))
            self.zoom_videos(os.path.join(video_path, gesture))

    def flip_videos(self, video_path:str):

        """ Horizontally Flip Videos """

        # Get the List of the Videos and Last Video Number
        video_list = [f for f in os.listdir(video_path) if f.endswith('.mp4') or f.endswith('.avi')]
        num_videos = len(video_list)

        # Loop on All Videos
        for i, video in enumerate(tqdm(natsorted(video_list))):

            # Open Video Cap
            cap = cv2.VideoCapture(os.path.join(video_path, video))

            # Get Video Info
            fps, num_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a VideoWriter Object
            out = cv2.VideoWriter(os.path.join(video_path, f"{num_videos+i+1}.avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), True)

            # Loop on Video Frames
            for _ in range(num_frames):

                # Get Frame
                ret, frame = cap.read()

                # Horizontally Flip Frame
                flipped_frame = cv2.flip(frame, 1)

                # Write Frame in Output Video
                out.write(flipped_frame)

            # Close VideoWriter and Cap
            out.release()
            cap.release()

    def zoom_videos(self, video_path:str, zoom_scale:float=0.9):

        """ 90% Zoom Videos """

        # Get the List of the Videos and Last Video Number
        video_list = [f for f in os.listdir(video_path) if f.endswith('.mp4') or f.endswith('.avi')]
        num_videos = len(video_list)

        # Loop on All Videos
        for i, video in enumerate(tqdm(natsorted(video_list))):

            # Open Video Cap
            cap = cv2.VideoCapture(os.path.join(video_path, video))

            # Get Video Info
            fps, num_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a VideoWriter Object
            out = cv2.VideoWriter(os.path.join(video_path, f"{num_videos+i+1}.avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), True)

            # Loop on Video Frames
            for _ in range(num_frames):

                # Get Frame
                ret, frame = cap.read()

                # Zoom and Crop Frame
                zoomed_frame  = cv2.resize(frame, (0, 0), fx=1/zoom_scale, fy=1/zoom_scale)
                cropped_frame = zoomed_frame[0:frame_height, 0:frame_width]

                # Write Frame in Output Video
                out.write(cropped_frame)

            # Close VideoWriter and Cap
            out.release()
            cap.release()

if __name__ == '__main__':

    # Create Flip and Zoom Dataset Class
    FlipZoomDataset()
