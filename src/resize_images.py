#import cv2
import tensorflow as tf
import numpy as np
import argparse
import glob
import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../common_train"))
from string_to_boolean import string_to_boolean

def resize_and_pad_image(animage, target_w, target_h, dopad=False):
    height, width, _ = animage.shape
    aspect_ratio = width / height
    new_width = target_w
    new_height = int(new_width / aspect_ratio)
    logging.debug("Width {}, Height {}, Aspect {}, New W {}, New H {}".format(width, height, aspect_ratio, new_width, new_height))
    #resized_image = cv2.resize(animage, (new_width, new_height))
    resized_image = tf.image.resize(animage, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if dopad:
        # Pad the image to the target size
        padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # This is for centering the image
        #x_offset = (target_w - new_width) // 2
        #y_offset = (target_h - new_height) // 2

        # This is for top left corner
        x_offset = 0
        y_offset = 0
        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_image

        return padded_image
    else:
        return resized_image

def main():
    
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'imagedir',
        help='Path of the image dir to resize.')
    parser.add_argument(
        '--outdir',
        help='output directory name relative to pwd, default is ./',
        required=False,
        default="./")
    parser.add_argument(
        '--newwidth',
        help='New width to scale image to',
        required=False,
        default=320)
    parser.add_argument(
        '--newheight',
        help='New width to scale image to',
        required=False,
        default=320)
    parser.add_argument(
        '--padimage',
        help='Pad image to full width height',
        default=False)

    args = parser.parse_args()
    image_path = args.imagedir
    new_width = int(args.newwidth)
    new_height = int(args.newheight)
    pad_image = args.padimage
    out_dir = args.outdir

    pad_image = string_to_boolean(pad_image)

    logging.basicConfig(format='%(asctime)s resize_annotations: %(levelname)s: %(message)s', level=logging.INFO, force=True)

    logging.info("Resize image with new width and height of {}x{} and padded {} from dir {}".format(new_width, new_height, pad_image, image_path))

    os.makedirs(out_dir, exist_ok=True)

    file_list = glob.glob("{}/*.jpg".format(image_path))
    count = 0    
    for file in file_list:
        # Read and resize the image
        #image = cv2.imread(file)
        raw = tf.io.read_file(file)
        image = tf.io.decode_jpeg(raw, channels=3)

        out_file = os.path.join(out_dir, os.path.basename(file))

        #cv2.imwrite(out_file, resize_and_pad_image(image, new_width, new_height, pad_image))
        aug_image = resize_and_pad_image(image, new_width, new_height, pad_image)
        write_image = tf.io.encode_jpeg(aug_image)
        tf.io.write_file(out_file, write_image)
        count += 1
        #print("Processed file {} to {}", file, out_file)
    logging.info("Processed {} files.".format(count))

if __name__ == '__main__':
    main()