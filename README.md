# phone-usage-computer-vision
Identify handheld phones (in hand, on lap, near face). Ignore unrelated objects.

# setup
You can build either the yolo 11 version or a tflite version

This setup instructions for python utilize Ancaonda or Miniconda

For the yolo version, use the instructions in yolo_cpu_setup.txt or yolo_gpu_setup.txt

For the tflite version:
  $ conda create --name tfbuildrun python=3.9
  $ ./tf_runtime_setup.sh

# model build

For the tflite version:
  $ conda activate tfbuildrun
  $ bin/retrain_tf_model.sh

For the yolo version
  $ conda activate [uiltralytics-gpu-env | ultralytics-cpu-env]
  $ bin/retrain_yolo_model.sh

# run test tool

For tflite verison:
  $ conda activate tfbuildrun
  $ python src/test_tf_video_detect.py --model workspace/runs/detect/train/weights/best.pt --sync video --max-height 720 --threshold .1 videos/training/20250718_150650_075a44fc.mp4

For yolo version:
  $ conda activate [uiltralytics-gpu-env | ultralytics-cpu-env]
  $ python src/test_yolo_video_detect.py --model workspace/runs/detect/train/weights/best.pt --sync video --max-height 1050 --threshold .45 videos/training/20250718_145802_46039155.mp4
