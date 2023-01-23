# MediaPipe Installation Instruction - Ubuntu 18.04

How to install mediapipe on Ubuntu 18.04

## Getting Started

Some quick references to the guides and to the official tutorials of MediaPipe.

### MediaPipe Repository:

- https://github.com/google/mediapipe

### Installation Tutorials:

- https://google.github.io/mediapipe/getting_started/install.html

- https://google.github.io/mediapipe/getting_started/cpp.html

### Dependencies

- Ubuntu 18.04
- Python 3.6+

## Installation

1. **Install Go**

    - Download the last Go version (we use 1.16.3) from the official [download page](https://golang.org/dl/) or using wget:

        `wget https://dl.google.com/go/go1.16.3.linux-amd64.tar.gz `

    - Go to the download folder, extract and install:

        `sudo tar -xvf go1.16.3.linux-amd64.tar.gz`

        `sudo mv go /usr/local`

    - Setup Go enviroment in file `.bashrc`:
        
      - GOROOT is the install location of Go.
      - GOPATH is the location of your work directory.

        `export GOROOT=/usr/local/go`
        
        `export GOPATH=$HOME/folder_where_Go_packages_will_be_saved`
        
        `export PATH=$GOPATH/bin:$GOROOT/bin:$PATH`

    - Verify Installation, type on terminal:

        `go version`

            go version go1.15.2 linux/amd64

        `go env`

            GOARCH="amd64"
            GOBIN=""
            GOCACHE="/root/.cache/go-build"
            GOEXE=""
            GOHOSTARCH="amd64"
            GOHOSTOS="linux"
            GOOS="linux"
            GOPATH="/root/Projects/Proj1"
            GORACE=""
            GOROOT="/usr/local/go"
            GOTMPDIR=""
            GOTOOLDIR="/usr/local/go/pkg/tool/linux_amd64"
            GCCGO="gccgo"
            CC="gcc"
            CXX="g++"
            CGO_ENABLED="1"
            ...
            ...
        
2. **Install Bazelisk**
   
    - Install with Go:

        > go get github.com/bazelbuild/bazelisk

    - Add to PATH in `.bashrc`

        `export PATH=$PATH:$(go env GOPATH)/bin`

    - Chech the installation by typing on terminal:

        `bazelisk help`

    - Add bazel alias to `.bash_aliases` file:

        `alias bazel='bazelisk'`

3. **Install gcc-8**

    - Install from binaries:

        `sudo apt-get install gcc-8 g++-8`

    - Set new default compiler:

        `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8`

    - Check gcc version (gcc8 = 8.4.0):

        `gcc --version`

4. **Clone MediaPipe repository**

    > git clone https://github.com/google/mediapipe.git
    
5. **Install OpenCV and FFmpeg**

    - Install dependancies from source:

        `sudo apt-get install libopencv-core-dev libopencv-highgui-dev libopencv-calib3d-dev libopencv-features2d-dev libopencv-imgproc-dev libopencv-video-dev`

        `pip3 install absl-py attrs dataclasses numpy opencv-contrib-python protobuf six wheel`


    - Configure MediaPipe Enviroment

        - Go into mediapipe installation folder

        **Ubuntu 18.04:**

        - Debian 9 and Ubuntu 18.04 install the packages in `/usr/lib/x86_64-linux-gnu`. MediaPipe’s `opencv_linux.BUILD` and `ffmpeg_linux.BUILD` are configured for this library path. 

        **Ubuntu 20.04:** 
        
        - Ubuntu 20.04 may install the OpenCV and FFmpeg packages in `/usr/local`. Follow the instructions below to modify the `WORKSPACE`, `opencv_linux.BUILD` and `ffmpeg_linux.BUILD` to point MediaPipe to your own OpenCV and FFmpeg libraries. For example if OpenCV and FFmpeg are both manually installed in `/usr/local/`, you will need to update:
          
             > The `"new_local_repository"` rules for `"linux_opencv"` and `"linux_ffmpeg"` in `"WORKSPACE"`.

            > The `"cc_library"` rule for `"opencv"` in `"opencv_linux.BUILD"`

            > The `"cc_library"` rule for `"libffmpeg"` in `"ffmpeg_linux.BUILD"`

        - An example below:

                new_local_repository(
                    name = "linux_opencv",
                    build_file = "@//third_party:opencv_linux.BUILD",
                    path = "/usr/local",
                )

                new_local_repository(
                    name = "linux_ffmpeg",
                    build_file = "@//third_party:ffmpeg_linux.BUILD",
                    path = "/usr/local",
                )

                cc_library(
                    name = "opencv",
                    srcs = glob(
                        [
                            "lib/libopencv_core.so",
                            "lib/libopencv_highgui.so",
                            "lib/libopencv_imgcodecs.so",
                            "lib/libopencv_imgproc.so",
                            "lib/libopencv_video.so",
                            "lib/libopencv_videoio.so",
                        ],
                    ),
                    hdrs = glob([
                        # For OpenCV 3.x
                        "include/opencv2/**/*.h*",
                        # For OpenCV 4.x
                        # "include/opencv4/opencv2/**/*.h*",
                    ]),
                    includes = [
                        # For OpenCV 3.x
                        "include/",
                        # For OpenCV 4.x
                        # "include/opencv4/",
                    ],
                    linkstatic = 1,
                    visibility = ["//visibility:public"],
                )

                cc_library(
                    name = "libffmpeg",
                    srcs = glob(
                        [
                            "lib/libav*.so",
                        ],
                    ),
                    hdrs = glob(["include/libav*/*.h"]),
                    includes = ["include"],
                    linkopts = [
                        "-lavcodec",
                        "-lavformat",
                        "-lavutil",
                    ],
                    linkstatic = 1,
                    visibility = ["//visibility:public"],
                )


6. **Install GPU acceleration dependancies**

    - Requires a GPU with EGL driver support.
    - Can use mesa GPU libraries for desktop, (or Nvidia/AMD equivalent):

        `sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev`

7. **Compile (Hello Word! Example)**

    - The build commands are the same as in the run, but it is important to build each code first to see any errors and because it will take some time.
    - Go into mediapipe installation folder.

    - **Build with GPU Support:**

        `bazelisk run --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/hello_world:hello_world`

    - **Build with Only CPU:**

        `bazelisk run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world`


## GPU Support

Configure the system to enable OpenGL and Tensorflow. [Here](https://google.github.io/mediapipe/getting_started/gpu_support.html) the official guide.

1. **OpenGL ES Support**

    MediaPipe supports OpenGL ES up to version 3.2 on Linux. On Linux desktop with video cards that support OpenGL ES 3.1+, MediaPipe can run GPU compute and rendering and perform TFLite inference on GPU.
    
    - Install dependancies:

        `sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev`

        `sudo apt-get install mesa-utils`
    
    - Grep OpenGL Info:

        `glxinfo | grep -i opengl`

            ...
            OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 430.50
            OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.20
            OpenGL ES profile extensions:

    - You need to see ES 3.1 or greater printed in order to perform TFLite inference on GPU in MediaPipe.

1. **CUDA Support**

    MediaPipe framework doesn’t require CUDA for GPU compute and rendering. However, MediaPipe can work with TensorFlow to perform GPU inference on video cards that support CUDA.

    - Install CUDA and cuDNN following the [official nvidia guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). (I used CUDA 11.0 Update 1 and cuDNN 8.1.1)

    - Useful Links:

        > [CUDA Installation Tutorial](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

        > [CUDA Download Archive](https://developer.nvidia.com/cuda-toolkit-archive)

        > [cuDNN Installation Tutorial](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

        > [cuDNN Download Archive](https://developer.nvidia.com/rdp/cudnn-archive)

    - Install [TensorFlow](https://github.com/tensorflow/tensorflow) and [onxx](https://pypi.org/project/onnx/):

        `pip3 install tensorflow onxx`

    - Install [PyTorch](https://pytorch.org/):

        `pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

    - Install pycuda:

        `pip3 install 'pycuda>=2019.1.1'`

    - Install the correct version of TensorRT from [here](https://developer.nvidia.com/tensorrt) (I used TensorRT 7.2.3 for Ubuntu 18.04 and CUDA 11.0)

        > [TensorRT Installation Tutorial](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

        > [TensorRT Download Archive](https://developer.nvidia.com/nvidia-tensorrt-7x-download)

    - Add Following Lines to `.bashrc`:

        `export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}`

        `export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64,/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
        
        `export TF_CUDA_PATHS=/usr/local/cuda-11.0,/usr/lib/x86_64-linux-gnu,/usr/include`

    - Run on terminal:

        `sudo ldconfig`

    - Edit the file `.bazelrc` in mediapipe installation directory addind the following lines:

            # This config refers to building with CUDA available. It does not necessarily
            # mean that we build CUDA op kernels.
            build:using_cuda --define=using_cuda=true
            build:using_cuda --action_env TF_NEED_CUDA=1
            build:using_cuda --crosstool_top=@local_config_cuda//crosstool:toolchain

            # This config refers to building CUDA op kernels with nvcc.
            build:cuda --config=using_cuda
            build:cuda --define=using_cuda_nvcc=true

    - Finally, build MediaPipe with TensorFlow GPU with two more flags `--config=cuda` and `--spawn_strategy=local`:

        `bazelisk build -c opt --config=cuda --spawn_strategy=local --define no_aws_support=true --copt -DMESA_EGL_NO_X11_HEADERS mediapipe/examples/desktop/object_detection:object_detection_tensorflow`


## MediaPipe in Python

MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt Python package. MediaPipe Python package is available on PyPI for Linux, macOS and Windows.

1. **Install Dependancies**

    `sudo apt install python3-dev`

    `sudo apt install -y protobuf-compiler`

2. **[OPTIONAL] Install and Use Python Virtual Enviroment**
    
    `sudo apt install python3-venv`

    `python3 -m venv mp_env && source mp_env/bin/activate`

3. **Install MediaPipe on pip**

    `pip3 install mediapipe`

4. **Install Requirements**

    `cd .../mediapipe`

    `pip3 install -r requirements.txt`

5. **Setup and Build**

    `cd .../mediapipe`

    `python3 setup.py gen_protos`

    `python3 setup.py install --link-opencv`
    
## Run Examples

MediaPipe support lot of pre-built examples:

- **Export GLOG for Running**:

    `export GLOG_logtostderr=1`

- **Hello Word!**

  - **Run with GPU Support:**

      `bazelisk run --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/hello_world:hello_world`

  - **Run with Only CPU:**

      `bazelisk run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world`

- **Face Detection**

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/face_detection:face_detection_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_gpu --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt`

- **Face Mesh**

    Official [link](https://google.github.io/mediapipe/solutions/face_mesh.html).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/face_mesh:face_mesh_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_gpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_mesh:face_mesh_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt`

- **Iris**

    Official [link](https://google.github.io/mediapipe/solutions/iris.html).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/iris_tracking:iris_tracking_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_gpu --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt`

- **Hand Tracking**

    Official [link](https://google.github.io/mediapipe/solutions/hands.html).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt`

- **Pose**

    Official [link](https://google.github.io/mediapipe/solutions/pose).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/pose_tracking:pose_tracking_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_gpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt`

<!-- - **Pose - Upper-Body Only**
  
    Official [link](https://google.github.io/mediapipe/solutions/pose).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/upper_body_pose_tracking:upper_body_pose_tracking_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/upper_body_pose_tracking/upper_body_pose_tracking_gpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/upper_body_pose_tracking_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/upper_body_pose_tracking:upper_body_pose_tracking_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/upper_body_pose_tracking/upper_body_pose_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/upper_body_pose_tracking_cpu.pbtxt` -->

- **Holistic**
  
    Official [link](https://google.github.io/mediapipe/solutions/holistic.html).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/holistic_tracking:holistic_tracking_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/holistic_tracking/holistic_tracking_gpu --calculator_graph_config_file=mediapipe/graphs/holistic_tracking/holistic_tracking_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/holistic_tracking:holistic_tracking_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/holistic_tracking/holistic_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt`

- **Selfie Segmentation**
  
    Official [link](https://google.github.io/mediapipe/solutions/selfie_segmentation.html).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/selfie_segmentation:selfie_segmentation_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/selfie_segmentation/selfie_segmentation_gpu --calculator_graph_config_file=mediapipe/graphs/selfie_segmentation/selfie_segmentation_gpu.pbtxt`

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/selfie_segmentation:selfie_segmentation_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/selfie_segmentation/selfie_segmentation_cpu --calculator_graph_config_file=mediapipe/graphs/selfie_segmentation/selfie_segmentation_cpu.pbtxt`

- **Hair Segmentation**

    Official [link](https://google.github.io/mediapipe/solutions/hair_segmentation.html).

    - **GPU Support**

        Build:

        `bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 mediapipe/examples/desktop/hair_segmentation:hair_segmentation_gpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_gpu --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_mobile_gpu.pbtxt`

    - **Only CPU**

        N/A

- **Object Detection**

    Official [link](https://google.github.io/mediapipe/solutions/object_detection.html).

    - **GPU Support**

        N/A

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt`

- **Box Tracking**

    Official [link](https://google.github.io/mediapipe/solutions/box_tracking.html).

    - **GPU Support**

        N/A

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_tracking:object_tracking_cpu`
        
        Run:

        `bazel-bin/mediapipe/examples/desktop/object_tracking/object_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt`

- **Instant Motion Tracking**

    Official [link](https://google.github.io/mediapipe/solutions/instant_motion_tracking.html)

    - **GPU Support**

        N/A

    - **Only CPU**

        N/A

- **Objectron**

    Official [link](https://google.github.io/mediapipe/solutions/objectron.html).

    - **GPU Support**

        N/A

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection_3d:objectron_cpu`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection_3d/objectron_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection_3d/objectron_desktop_cpu.pbtxt --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>,box_landmark_model_path=<landmark model path>,allowed_labels=<allowed labels>`

- **KNIFT**

    Official [link](https://google.github.io/mediapipe/solutions/knift.html).

    - **GPU Support**

        N/A

    - **Only CPU**

        N/A

- **AutoFlip**

    Official [link](https://google.github.io/mediapipe/solutions/autoflip.html).

    - **GPU Support**

        N/A

    - **Only CPU**

        Build:

        `bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/autoflip:run_autoflip`
        
        Run:

        `GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip --calculator_graph_config_file=mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt --input_side_packets=input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/absolute/path/to/save/the/output/video/file,aspect_ratio=1:1`

- **MediaSequence**

    Official [link](https://google.github.io/mediapipe/solutions/media_sequence.html).

    - **GPU Support**

        N/A

    - **Only CPU**

        Build:

        `bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo --define MEDIAPIPE_DISABLE_GPU=1`

