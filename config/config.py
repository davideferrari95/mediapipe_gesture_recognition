from dataclasses import dataclass

@dataclass
class Params:

  # Enable Mediapipe Solutions
  enable_right_hand:    bool
  enable_left_hand:     bool
  enable_pose:          bool
  enable_face:          bool

  # Training Parameters
  min_epochs:           int
  max_epochs:           int
  patience:             int
  batch_size:           int
  optimizer:            str
  learning_rate:        float
  loss_function:        str
  train_set_size:       float
  validation_set_size:  float
  test_set_size:        float

  # Torch Compilation
  compilation_mode:     str
  torch_compilation:    bool
  fast_dev_run:         bool
