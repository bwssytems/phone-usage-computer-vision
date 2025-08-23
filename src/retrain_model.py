import os
import logging
import random
import time

from string_to_boolean import string_to_boolean
from create_path import create_path

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf

assert tf.__version__.startswith('2')

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import object_detector

def copyfile(src, dest):
    os.symlink(src, dest)

def split_dataset(images_path, annotations_path, val_split, test_split, out_path):
    """Splits a directory of sorted images/annotations into training, validation, and test sets.

    Args:
      images_path: Path to the directory with your images (JPGs).
      annotations_path: Path to a directory with your VOC XML annotation files,
        with filenames corresponding to image filenames. This may be the same path
        used for images_path.
      val_split: Fraction of data to reserve for validation (float between 0 and 1).
      test_split: Fraction of data to reserve for test (float between 0 and 1).
    Returns:
      The paths for the split images/annotations (train_dir, val_dir, test_dir)
    """
    _, dirs, _ = next(os.walk(images_path))

    train_dir = os.path.join(out_path, 'train')
    val_dir = os.path.join(out_path, 'validation')
    test_dir = os.path.join(out_path, 'test')

    IMAGES_TRAIN_DIR = os.path.join(train_dir, 'images')
    IMAGES_VAL_DIR = os.path.join(val_dir, 'images')
    IMAGES_TEST_DIR = os.path.join(test_dir, 'images')
    os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
    os.makedirs(IMAGES_TEST_DIR, exist_ok=True)

    ANNOT_TRAIN_DIR = os.path.join(train_dir, 'annotations')
    ANNOT_VAL_DIR = os.path.join(val_dir, 'annotations')
    ANNOT_TEST_DIR = os.path.join(test_dir, 'annotations')
    os.makedirs(ANNOT_TRAIN_DIR, exist_ok=True)
    os.makedirs(ANNOT_VAL_DIR, exist_ok=True)
    os.makedirs(ANNOT_TEST_DIR, exist_ok=True)

    # Get all filenames for this dir, filtered by filetype
    filenames = os.listdir(os.path.join(images_path))
    filenames = [os.path.join(images_path, f) for f in filenames if (f.endswith('.jpg'))]
    # Shuffle the files, deterministically
    filenames.sort()
    random.seed(42)
    random.shuffle(filenames)
    # Get exact number of images for validation and test; the rest is for training
    val_count = int(len(filenames) * val_split)
    test_count = int(len(filenames) * test_split)
    for i, file in enumerate(filenames):
        source_dir, filename = os.path.split(file)
        annot_file = os.path.join(annotations_path, filename.replace(".jpg", ".xml"))
        if i < val_count:
            copyfile(file, os.path.join(IMAGES_VAL_DIR, filename))
            copyfile(annot_file, os.path.join(ANNOT_VAL_DIR, filename.replace(".jpg", ".xml")))
        elif i < val_count + test_count:
            copyfile(file, os.path.join(IMAGES_TEST_DIR, filename))
            copyfile(annot_file, os.path.join(ANNOT_TEST_DIR, filename.replace(".jpg", ".xml")))
        else:
            copyfile(file, os.path.join(IMAGES_TRAIN_DIR, filename))
            copyfile(annot_file, os.path.join(ANNOT_TRAIN_DIR, filename.replace(".jpg", ".xml")))
    return (train_dir, val_dir, test_dir)

def main():
    logging.basicConfig(format='%(asctime)s retrain_model: %(levelname)s: %(message)s', level=logging.INFO, force=True)

    logging.info("----------------------------------------------")
    logging.info(f"   Start Training for eff def lite")
    logging.info("----------------------------------------------")

    max_detections = 25
    max_instances = 100
    label_map = {1: 'blank', 2: 'phone'} 
    
    logging.info(f"Using max instances: {max_instances} and max detections: {max_detections} and label_map: {label_map}")

    base_path = os.path.abspath(os.path.join(os.getcwd(), "workspace"))
    logging.info(f"Using path: {base_path}")

    split_dir_path = os.path.join(base_path, 'split-dataset')
    logging.info("split_dir_path: {}".format(split_dir_path))
    if os.path.exists(split_dir_path) == False:
        # If it's NOT split yet, specify the path to all images and annotations
        images_in = os.path.join(base_path, 'resize_images')
        annotations_in = os.path.join(base_path, 'resize_annotations')
        logging.info("Create Split dataset from images path: {}, annotations path: {}".format(images_in, annotations_in) )
        # We need to instantiate a separate DataLoader for each split dataset

        train_dir, val_dir, test_dir = split_dataset(images_in, annotations_in,
                                                    val_split=0.05, test_split=0.01,
                                                    out_path=split_dir_path)
    else:
        train_dir = os.path.join(split_dir_path, 'train')
        val_dir = os.path.join(split_dir_path, 'validation')
        test_dir = os.path.join(split_dir_path, 'test')
        logging.info("Using existing split dataset, train: {}, validation: {}, test: {}".format(train_dir, val_dir, test_dir))

    retrain_epochs = int(os.environ.get('RETRAIN_EPOCHS', 2))
    logging.info(f"retrain_epochs: {retrain_epochs}")
    retrain_batch_size = int(os.environ.get('RETRAIN_BATCH_SIZE', 32))
    logging.info(f"retrain_batch_size: {retrain_batch_size}")
    num_shards_usage = 100
    logging.info(f"num_shards_usage: {num_shards_usage}")

    train_data = object_detector.DataLoader.from_pascal_voc(
        os.path.join(train_dir, 'images'),
        os.path.join(train_dir, 'annotations'), label_map=label_map, num_shards=num_shards_usage)
    validation_data = object_detector.DataLoader.from_pascal_voc(
        os.path.join(val_dir, 'images'),
        os.path.join(val_dir, 'annotations'), label_map=label_map, num_shards=num_shards_usage)
    test_data = object_detector.DataLoader.from_pascal_voc(
        os.path.join(test_dir, 'images'),
        os.path.join(test_dir, 'annotations'), label_map=label_map, num_shards=num_shards_usage)

    logging.info(f'Start training with train count: {len(train_data)}, validation count: {len(validation_data)}, test count: {len(test_data)}')

    TFLITE_FILENAME = 'efficientdet-lite.tflite'
    LABELS_FILENAME = 'efficientdet-lite_labels.txt'
    tflite_file_path = os.path.join(base_path, TFLITE_FILENAME)

    use_multi_gpus = string_to_boolean(os.environ.get('USE_MULTI_GPUS', 'False'))
    logging.info(f"use_multi_gpus: {use_multi_gpus}")
    if use_multi_gpus:
        train_strategy = 'gpus'

        logging.info(f'Physical GPU Devices: {tf.config.list_physical_devices("GPU")}')
        logging.info(f'Logical GPU Devices: {tf.config.list_logical_devices("GPU")}')
    else:
        train_strategy = None

    train_whole_model_flag = string_to_boolean(os.environ.get('TRAIN_WHOLE_MODEL', 'True'))

    model_type = os.environ.get('MODEL_TYPE', 'efficientdet_lite0')
    if model_type is None:
        logging.error("MODEL_TYPE environment variable not set")
        exit(1)
    if model_type not in ['efficientdet_lite0', 'efficientdet_lite1', 'efficientdet_lite2', 'efficientdet_lite3', 'efficientdet_lite3x', 'efficientdet_lite4']:
        logging.error("MODEL_TYPE environment variable not set to a valid model type")
        exit(1)

    logging.info(f"Using model type: {model_type}")

    hparams = ''

    tf.get_logger().setLevel('ERROR')

    # Create a directory with current date and time
    tf_model_path = create_path(base_path,'tf_model', False, logging)
    use_xla_flag = False # It seems xla is not supported in TFlite

    if model_type == 'efficientdet_lite1':
        spec = object_detector.EfficientDetLite1Spec(hparams=hparams, model_dir=tf_model_path, strategy=train_strategy, tflite_max_detections=max_detections, use_xla=use_xla_flag)
    elif model_type == 'efficientdet_lite2':
        spec = object_detector.EfficientDetLite2Spec(hparams=hparams, model_dir=tf_model_path, strategy=train_strategy, tflite_max_detections=max_detections, use_xla=use_xla_flag)
    elif model_type == 'efficientdet_lite3':
        spec = object_detector.EfficientDetLite3Spec(hparams=hparams, model_dir=tf_model_path, strategy=train_strategy, tflite_max_detections=max_detections, use_xla=use_xla_flag)
    elif model_type == 'efficientdet_lite3x':
        spec = object_detector.EfficientDetSpec(
            model_name='efficientdet-lite3x',
            uri = 'https://tfhub.dev/tensorflow/efficientdet/lite3x/feature-vector/1',
            hparams=hparams, model_dir=tf_model_path, strategy=train_strategy, tflite_max_detections=max_detections, use_xla=use_xla_flag)
    elif model_type == 'efficientdet_lite4':
        spec = object_detector.EfficientDetLite4Spec(hparams=hparams, model_dir=tf_model_path, strategy=train_strategy, tflite_max_detections=max_detections, use_xla=use_xla_flag)
    else:
        spec = object_detector.EfficientDetLite0Spec(hparams=hparams, model_dir=tf_model_path, strategy=train_strategy, tflite_max_detections=max_detections, use_xla=use_xla_flag)

    spec.config.max_instances_per_image = max_instances
    #spec.config.learning_rate = 0.001
    #spec.config.lr_warmup_init = 0.001
    #spec.config.autoaugment_policy = 'v2'
    #spec.config.optimizer = 'adam'

    logging.info(f"Using model spec config: {spec.config}")

    logging.info("Running TFLite Model Maker Object Detector create...")

    model = object_detector.create(train_data=train_data, 
                                model_spec=spec, 
                                validation_data=validation_data, 
                                epochs=retrain_epochs,
                                batch_size=retrain_batch_size,
                                train_whole_model=train_whole_model_flag)

    model.summary()

    export_dir = create_path(base_path,'tf_model_export', False, logging)
    logging.info(f"Exporting model to: {export_dir}")
    use_quant_int8 = string_to_boolean(os.environ.get('USE_QUANT_INT8', 'False'))
    if use_quant_int8:
        logging.info("Using Quantization for int8")
        config = QuantizationConfig.for_int8(test_data)
        model.export(export_dir=export_dir, tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME, quantization_config=config,
                     export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
    else:
        model.export(export_dir=export_dir, tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
                    export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

    try:
        logging.info("Evaluating model")
        result = model.evaluate(test_data)
        logging.info(f"Evaluate model Result: {result}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")

    use_eval_tflite = string_to_boolean(os.environ.get('USE_EVAL_TFLITE', 'False'))
    if use_eval_tflite:
        try:
            logging.info("Evaluating tflite model")
            result = model.evaluate_tflite(tflite_file_path, test_data)
            logging.info(f"Evaluate tflite model Result: {result}")
        except Exception as e:
            logging.error(f"Error evaluating tflite model: {e}")

    logging.info(f"\nRecap of training paramters:")
    logging.info(f"  Using max instances: {max_instances} and max detections: {max_detections} and label_map: {label_map}")
    logging.info(f"  Using path: {base_path}")
    logging.info(f'  Start training with train count: {len(train_data)}, validation count: {len(validation_data)}, test count: {len(test_data)}')
    logging.info(f"  retrain_epochs: {retrain_epochs}")
    logging.info(f"  retrain_batch_size: {retrain_batch_size}")
    logging.info(f"  num_shards_usage: {num_shards_usage}")
    logging.info(f"  use_multi_gpus: {use_multi_gpus}")
    logging.info(f"  Using model type: {model_type}")
    logging.info(f"  Using model spec config: {spec.config}")

    logging.info("\n")
    logging.info(f"----------- Retrain Complete -----------")

if __name__ == "__main__":
    main()
    time.sleep(10)
    exit(0)