#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Convert YOLO txt labels to LabelMe JSON.

From input path read images from "images" and labels from "labels" directories.

Example:
    ./input_dir
    |-> /images/*.jpg
    `-> /labels/*.txt

1. create output dir
2. copy images to output dir and create json files

example:
./convert_yolo2labelme.py --input=./res --output=./res_json --classes=./res/class_names.txt

'''

__author__ = 'Vladimir'

import os
import json
import cv2
import glob
import argparse
import shutil
import base64
import numpy as np
import importlib

try:
    import labelme
except ModuleNotFoundError:
    is_labelme_available = False


def check_module_available(module_name):
    module_spec = importlib.util.find_spec(module_name)
    found = module_spec is not None
    return found

def get_base64_from_image(img):
    # return base64 data from compressed to JPEG image
    if img is None:
        return None
    
    # set encode param
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # compress image into buffer
    result, imgencode = cv2.imencode(".jpg", img, encode_param)
    #print(result, type(imgencode))
    
    data = np.array(imgencode)
    #print(data.shape, data.dtype)
    # base64 encode
    img_data = base64.b64encode(data)
    #print(len(img_data), type(img_data))
    res = img_data.decode("ascii")
    return res

def convert_yolo_labels_to_shapes(labels, image, class_names):
    """
    convert yolo labels to labelme shapes
    """
    shapes = []

    if labels is None or image is None:
        return None

    image_height, image_width = image.shape[:2]

    # iterate through labels
    for row in labels:
        print(row)
        # prepare list of shapes

        res = row.split()
        # class_idx x_center y_center width height
        print(res)
        class_id = int(res[0])
        box = ( float(res[1])*image_width, float(res[2])*image_height, float(res[3])*image_width, float(res[4])*image_height)
        w_2 = box[2]/2
        h_2 = box[3]/2
        
        x1, y1 = box[0]-w_2, box[1]-h_2
        x2, y2 = box[0]+w_2, box[1]+h_2
        points = [(x1, y1), (x2, y2)]
        label = class_names[class_id]
        shape = dict(
            label=label,
            points=points,
            group_id=None,
            shape_type="rectangle",
            flags={},
        )
        shapes.append(shape)
    return shapes

def save_data_to_json(
        is_labelme_available,
        image_filename,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth):
    # save data to JSON via labelme functions or by hands
    if is_labelme_available:
        # save data via labelme
        imageData = labelme.LabelFile.load_image_file(image_filename)
        labelFile = labelme.LabelFile()

        try:
            labelFile.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=imageHeight,
                imageWidth=imageWidth)
        except Exception as e:
            print("Error write json-file:", filename)
            raise e
    else:
        # save data manually

        # set fields
        dataj = {}
        dataj["version"] = None
        dataj["flags"] = {}
        dataj["shapes"] = shapes
        dataj["imagePath"] = imagePath
        dataj["imageData"] = None # get_base64_from_image(image)
        dataj["imageHeight"] = imageHeight
        dataj["imageWidth"] = imageWidth

        # print(dataj)

        try:
            # save JSON data into file
            with open(filename, "w") as fs:
                json.dump(dataj, fs, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Error write file:", filename)
            raise e

    return 0

def main():

    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="input", default="./res", type=str,
                    help="path to directory with images and YOLO txt files")
    ap.add_argument("-o", "--output", required=True, dest="output", default="./res_json", type=str,
                    help="path to directory for store results")
    ap.add_argument("-c", "--classes", dest="classes", default="./res/class_names.txt", type=str,
                    help="file with class labels")
    ap.add_argument("-e", "--extention", dest="extention", default="jpg", type=str,
                    help="extention of image files")
    args = vars(ap.parse_args())
    # print(args)

    print("Start...")

    # check extention
    ext = args["extention"]
    if ext.startswith("*"):
        ext = ext[1:]
    if ext.startswith("."):
        ext = ext[1:]
    print("Image extention:", ext)

    # check - is labelme available?
    is_labelme_available = False
    if check_module_available("labelme"):
        is_labelme_available = True
        print("labelme found!")
    else:
        print("labelme not found!")

    # get image files
    get_images_path = os.path.join(args["input"], "images", "*."+ext)
    print("Get images:", get_images_path)
    files = sorted(glob.glob(get_images_path))
    print("Images files:", files)

    # read classes from file
    with open(args["classes"]) as fs:
        classes = fs.read().splitlines()
    print("class_names:", classes)

    dst_dir = args["output"]
    # create output dir
    os.makedirs(dst_dir, exist_ok=True)

    for filename in files:
        print("image:", filename)
        # convert image path to label path
        sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep  # /images/, /labels/ substrings
        image_name = os.path.basename(filename)
        image_basename = os.path.splitext(image_name)[0]
        labels_name = image_basename+".txt"
        labels_filename = filename.replace(sa, sb, 1)
        labels_filename = labels_filename.replace(image_name, labels_name, 1)
        print("labels: ", labels_filename)

        print("Read", filename)
        image = cv2.imread(filename)
        if image is None:
            print("[!] Error read image:", filename)
            continue
        
        height, width = image.shape[:2]
        print("Image size:", width, height)

        # copy image
        dst_image_path = os.path.join(dst_dir, image_name)
        print("Copy image to:", dst_image_path)
        shutil.copyfile(filename, dst_image_path)

        shapes = None
        with open(labels_filename, "r") as fs:
            labels = fs.readlines()
            print(labels)

            shapes = convert_yolo_labels_to_shapes(labels, image, classes)
        
        if len(shapes) == 0:
            print("No shapes for save!")
            continue

        res_file_name = image_basename + ".json"
        dst_file_path = os.path.join(dst_dir, res_file_name)

        # save to JSON-file
        save_data_to_json(is_labelme_available=is_labelme_available,
                          image_filename=filename,
                          filename=dst_file_path,
                          shapes=shapes,
                          imagePath=image_name,
                          imageHeight=height,
                          imageWidth=width)

    print("Done.")


if __name__ == '__main__':
    main()
