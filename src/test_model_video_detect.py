#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import time

import tflite_runtime.interpreter as tflite

tflite_model = "./model.tflite"
video_path = "0"
det_thresh = 0.5

DETECTION_THRESHOLD = 0.5

# Define a list of colors for visualization when there are only 3 classes
COLORS = ['???'] * 3
COLORS[0] = (0,255,0)
COLORS[1] = (255,0,0)
COLORS[2] = (0,0,255)

# Define a list of colors for visualization where there are more than 3 classes
colours = np.random.randint(0, 255, size=(32, 3), dtype=np.uint8)

labels_array = []
filters_array = []
scores_index = 0
boxes_index = 1
classes_index = 3
count_index = 2
filters_avail = False
vidout = None
output_video = False

def load_labels(label_path):
  global labels_array
  with open(label_path) as my_file:
    labels_array = my_file.readlines()

  i = 0
  for labels in labels_array:
    labels_array[i] = labels.rstrip()
    i = i + 1
  print(labels_array)

def load_filters(filter_path):
  global filters_array
  global filters_avail
  if filter_path != None:
    with open(filter_path) as my_file:
      filters_array = my_file.readlines()

    i = 0
    for filters in filters_array:
      filters_array[i] = filters.rstrip()
      i = i + 1
    print(filters_array)
    filters_avail = True
  else:
    filters_avail = False

def not_filtered(filter_val):
  global filters_array
  global filters_avail
  if filters_avail != True:
    return True

  for aval in filters_array:
    if int(aval) == int(filter_val):
      return True
  return False

def gen_frames():
  """Continuously run inference on images acquired from the source.

  Args:
    model: Name of the TFLite object detection model.
    video_path: The video source to be passed to OpenCV.
    det_thresh: threshhold of score to label and use
  """

  global tflite_model
  global video_path
  global det_thresh
  global vidout
  global output_video

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  if video_path.isdigit():
    cap = cv2.VideoCapture(int(video_path))
  else:
    cap = cv2.VideoCapture(video_path)

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

  # Visualization parameters
  row_size = 40  # pixels
  text_x = 760  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 2
  font_thickness = 2
  fps_avg_frame_count = 15

  # Load TFLite model and allocate tensors.
  interpreter = tflite.Interpreter(model_path=tflite_model)
  interpreter.allocate_tensors()

  while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      cap.release()
      break    # Run inference and draw detection result on the local copy of the original file

    counter += 1

    detection_result_image = run_odt_and_draw_results(
        frame, 
        interpreter, 
        threshhold=float(det_thresh)
    )

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (text_x, row_size)
    cv2.putText(detection_result_image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    ret, buffer = cv2.imencode('.jpg', detection_result_image)
    frame = buffer.tobytes()
    if output_video:
      #if counter%20 == 0:
      #  main_logger.info("Current counter is %d"%counter)
      vidout.write(detection_result_image)

    cv2.imshow("Detection Result",detection_result_image)
    # Wait for a keystroke.   
    if cv2.waitKey(1) == ord('q'):
      break

  cap.release()

def preprocess_image(image, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  original_image = image
  resized_img = cv2.resize(image, input_size)
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Return the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshhold):
  """Returns a list of detection results, each a dictionary of object info."""
  global scores_index
  global classes_index
  global boxes_index
  global count_index

  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, int(scores_index))
  boxes = get_output_tensor(interpreter, int(boxes_index))
  count = int(get_output_tensor(interpreter, int(count_index)))
  classes = get_output_tensor(interpreter, int(classes_index))

  results = []
  for i in range(count):
    if not_filtered(classes[i]) and scores[i] >= threshhold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results

def run_odt_and_draw_results(image, interpreter, threshhold=0.3):
  """Run object detection on the input image and draw the detection results"""
  global labels_array

  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # print("Input height %s, Input width %s"%(input_height,input_width))
  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image, 
      (input_width, input_height)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshhold=threshhold)
  # print(results)
  # Plot the detection results on the input image
  detections = []  # np.array([])
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin_a, xmin_a, ymax_a, xmax_a = obj['bounding_box']
    element = []  # np.array([])
    element.append(xmin_a)
    element.append(ymin_a)
    element.append(xmax_a)
    element.append(ymax_a)
    element.append(obj['score'])  # print('element= ',element)
    detections.append(element)  # print('dets: ',dets)

    # Find the class index of the current object
    class_id = int(obj['class_id'])

  detections = np.array(detections)
  trdata = []
  if detections.any():

    for obj in results:
      # Convert the object bounding box from relative coordinates to absolute 
      # coordinates based on the original image resolution
      ymin, xmin, ymax, xmax = obj['bounding_box']
      xmin = int(xmin * original_image.shape[1])
      xmax = int(xmax * original_image.shape[1])
      ymin = int(ymin * original_image.shape[0])
      ymax = int(ymax * original_image.shape[0])
      # Draw the bounding box and label on the image
      # Find the class index of the current object
      class_id = int(obj['class_id'])
      if len(labels_array) == 2:
        color = [int(c) for c in COLORS[class_id]]
      else:
        color_index = class_id%32
        color = [int(c) for c in colours[color_index,:]]
      cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color, 2)
      # Make adjustments to make the label visible for all objects
      y = ymin - 15 if ymin - 15 > 15 else ymin + 15
      label = "{}: {:.0f}%".format(labels_array[class_id], obj['score'] * 100)
      cv2.putText(original_image, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image.astype(np.uint8)
  return original_uint8

def get_args():
  global tflite_model
  global video_path
  global det_thresh
  global scores_index
  global classes_index
  global boxes_index
  global count_index
  global vidout
  global output_video

  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      'video',
      help='Path of the video for detection.')
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='./model.tflite')
  parser.add_argument(
      '--labels',
      help='Path to labels for the object detection model.',
      required=False,
      default='./labels.txt')
  parser.add_argument(
      '--filter',
      help='Path to filter items for the object detection model.')
  parser.add_argument(
      '--videoout',
      help='Path of the outut video.',
      required=False)
  parser.add_argument(
      '--threshold',
      help='Threshold value for the object detection model.',
      required=False,
      default=0.1)
  parser.add_argument(
      '--scoresidx',
      help='Scores output tensor index for the object detection model.',
      required=False,
      default=0)
  parser.add_argument(
      '--classesidx',
      help='Classes output tensor index for the object detection model.',
      required=False,
      default=3)
  parser.add_argument(
      '--boxesidx',
      help='Boxes output tensor index for the object detection model.',
      required=False,
      default=1)
  parser.add_argument(
      '--countidx',
      help='Count output tensor index for the object detection model.',
      required=False,
      default=2)

  args = parser.parse_args()

  tflite_model, labels_file, filters_file, video_path, det_thresh = args.model, args.labels, args.filter, args.video, args.threshold
  video_out_path = args.videoout

  scores_index, classes_index, boxes_index, count_index = args.scoresidx, args.classesidx, args.boxesidx, args.countidx
  load_labels(labels_file)
  load_filters(filters_file)

  if video_out_path is None:
    output_video = False
  else:
    output_video = True

  if output_video:
    video_out_path = os.path.abspath(video_out_path)
    print("Video output path: %s"%video_out_path)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    vidout = cv2.VideoWriter(video_out_path, fourcc, 9.0, (int(960),int(720)))

  gen_frames()

if __name__ == '__main__':
  get_args()
