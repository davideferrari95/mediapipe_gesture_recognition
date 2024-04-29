import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark
from mediapipe.tasks.python.components.containers.detections import DetectionResult
from mediapipe.python.solutions.objectron import ObjectronOutputs

@dataclass
class LandmarkList:

    """ Landmark List Dataclass """

    landmark: List[Landmark]

@dataclass
class NormalizedLandmarkList:

    """ Normalized Landmark List Dataclass """

    landmark: List[NormalizedLandmark]

@dataclass
class HolisticResults:

    """ Holistic Result Dataclass

    Fields:

        pose_landmarks:       field that contains the pose landmarks.
        pose_world_landmarks: field that contains the pose landmarks in real-world 3D coordinates that are in meters with the origin at the center between hips.
        left_hand_landmarks:  field that contains the left-hand landmarks.
        right_hand_landmarks: field that contains the right-hand landmarks.
        face_landmarks:       field that contains the face landmarks.
        segmentation_mask:    field that contains the segmentation mask if "enable_segmentation" is set to true.

    """

    pose_landmarks:       Optional[NormalizedLandmarkList] = None
    pose_world_landmarks: Optional[LandmarkList]           = None
    left_hand_landmarks:  Optional[NormalizedLandmarkList] = None
    right_hand_landmarks: Optional[NormalizedLandmarkList] = None
    face_landmarks:       Optional[NormalizedLandmarkList] = None
    segmentation_mask:    Optional[np.ndarray]             = None

@dataclass
class ObjectronResults:

    """ Objectron Result Dataclass

    Fields:

        detected_objects: field that contains a list of detected 3D bounding boxes. Each detected box is represented as an "ObjectronOutputs" instance.

    """

    detected_objects: List[ObjectronOutputs]
