import xml.etree.ElementTree as ET
import argparse
import glob
import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../common_train"))
from string_to_boolean import string_to_boolean

def resize_annotation(target_shape, tree, dopad, file):
    root = tree.getroot()

    # Calculate the scaling factors
    original_width = int(root.find('size/width').text)
    original_height = int(root.find('size/height').text)
    target_width,target_height = target_shape
    
    try:
        target_aspect = target_width / target_height

        original_aspect = original_width / original_height
        
        new_width = target_width
        width_scale = new_width / original_width
        if target_aspect != original_aspect:
            if dopad:
                new_height = int(new_width / original_aspect)
                height_scale = new_height / original_height
            else:
                new_height = int(new_width / target_aspect)
                height_scale = new_height / original_height
        else:
            new_height = int(new_width / original_aspect)
            height_scale = new_height / original_height
    except ZeroDivisionError:
        logging.error("Zero division error for file {}".format(file))
        return None
    
    logging.debug("Width scale {} and height scale {}".format(width_scale, height_scale))
    if dopad:
        # This is for centering the image
        #x_offset = (target_width - new_width) // 2
        #y_offset = (target_height - new_height) // 2

        # This is for top left corner
        x_offset = 0
        y_offset = 0
        #print("Do padding with x_offset {} and y_offset {}".format(x_offset,y_offset))
    else:
        x_offset = 0
        y_offset = 0

    # Update the size information
    root.find('size/width').text = str(target_width)
    root.find('size/height').text = str(target_height)

    # Update the bounding box information
    for obj in root.findall('object'):
        class_name = obj.find('name')
        class_text = class_name.text
        class_name.text = class_text.lower()
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        xmin_new = int(round(xmin * width_scale))
        ymin_new = int(round(ymin * height_scale))
        xmax_new = int(round(xmax * width_scale))
        ymax_new = int(round(ymax * height_scale))
        bbox.find('xmin').text = str(xmin_new + x_offset)
        bbox.find('ymin').text = str(ymin_new + y_offset)
        bbox.find('xmax').text = str(xmax_new + x_offset)
        bbox.find('ymax').text = str(ymax_new + y_offset)
    return tree

def main():

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'annotedir',
        help='Path of the annotation dir to resize.')
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
    annote_path = args.annotedir
    new_width = int(args.newwidth)
    new_height = int(args.newheight)
    pad_image = args.padimage
    out_dir = args.outdir

    pad_image = string_to_boolean(pad_image)

    logging.basicConfig(format='%(asctime)s resize_annotations: %(levelname)s: %(message)s', level=logging.INFO, force=True)

    logging.info("Resize xml with new width and height of {}x{} and padded {} from dir {}".format(new_width, new_height, pad_image, annote_path))

    os.makedirs(out_dir, exist_ok=True)

    file_list = glob.glob("{}/*.xml".format(annote_path))
    count = 0
    for file in file_list:
        # Read and resize the image
        tree = ET.parse(file)

        out_file = os.path.join(out_dir, os.path.basename(file))

        tree = resize_annotation((new_width,new_height), tree, pad_image, file)
        count += 1
        if tree is None:
            continue

        tree.write(out_file)
        #print("Processed file {} to {}", file, out_file)
    logging.info("Processed {} files.".format(count))

if __name__ == '__main__':
    main()
