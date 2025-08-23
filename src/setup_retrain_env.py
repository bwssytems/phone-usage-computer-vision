import os
import logging
import argparse

from create_path import create_path
from exec_command import exec_command

def copyfile(src, dest):
    os.symlink(src, dest)

def main():
    global test_flag

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--rootpath',
        help='Root path to utilities',
        required=False,
        default=os.getcwd())
    parser.add_argument(
        '--datapath',
        help='Path to the dataset',
        required=False,
        default=os.path.join(os.getcwd(), 'data'))
    args = parser.parse_args()
    root_path = args.rootpath
    data_path = args.datapath

    logging.basicConfig(format='%(asctime)s setup_retrain_env: %(levelname)s: %(message)s', level=logging.INFO, force=True)

    logging.info(f"Setup for retraining the model")

    model_type = os.environ.get('MODEL_TYPE', 'efficientdet_lite0')
    if model_type is None:
        logging.error("MODEL_TYPE environment variable not set")
        exit(1)
    if model_type not in ['efficientdet_lite0', 'efficientdet_lite1', 'efficientdet_lite2', 'efficientdet_lite3', 'efficientdet_lite3x', 'efficientdet_lite4']:
        logging.error("MODEL_TYPE environment variable not set to a valid model type: {}".format(model_type))
        exit(1)

    do_resize = os.environ.get('DO_RESIZE', 'True')
    test_flag = False
    do_pad = 'False'

    model_width = 320
    model_height = 320
    if model_type == 'efficientdet_lite1':
        model_width = 384
        model_height = 384
    elif model_type == 'efficientdet_lite2':
        model_width = 448
        model_height = 448
    elif model_type == 'efficientdet_lite3':
        model_width = 512
        model_height = 512
    elif model_type == 'efficientdet_lite3x':
        model_width = 640
        model_height = 640
    elif model_type == 'efficientdet_lite4':
        model_width = 640
        model_height = 640

    # Create working directory
    cwd_path = os.getcwd()

    # Create a directory with current date and time
    out_path = create_path(cwd_path, "workspace", test_flag, logging)

    logging.info("We will be processing in this directory: {}".format(out_path))

    logging.info(f"Model type: {model_type}")
    logging.info(f"Model width: {model_width}")
    logging.info(f"Model height: {model_height}")

    resize_images_obj_path = create_path(out_path,'resize_images', test_flag, logging)

    resize_annotations_obj_path = create_path(out_path,'resize_annotations', test_flag, logging)

    if do_resize:
        images_path = os.path.join(data_path, 'images')
        annotations_path = os.path.join(data_path, 'annotations')
        logging.info("Resize Images")
        exec_command(f"python3 {root_path}/src/resize_images.py {images_path} --outdir {resize_images_obj_path} --padimage {do_pad} --newwidth {model_width} --newheight {model_height}", test_flag, logging)

        logging.info("Resize Annotations")
        exec_command(f"python3 {root_path}/src/resize_annotations.py {annotations_path} --outdir {resize_annotations_obj_path} --padimage {do_pad} --newwidth {model_width} --newheight {model_height}", test_flag, logging)

if __name__ == "__main__":
    main()
    exit(0)
