#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
convert LabelMe JSON to YOLO txt

1. create dirs output, output/images and output/labels,
2. copy images with json file to output/images directory,
3. create file_name.txt in output/labels directory with YOLO format.

tested for:
labelme 3.16.7, 4.6.0, 5.0.1

example:
./convert_labelme2yolo.py --input=./t8 --output=./res --classes=./class_names.txt

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

# box form[x,y,w,h]


def convert(size, box):
    # x_center y_center width height
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) * dw / 2
    y = (box[1] + box[3]) * dh / 2
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)


def main():

    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="input", default="./4/", type=str,
                    help="path to directory with images and LabelMe JSON files")
    ap.add_argument("-o", "--output", required=True, dest="output", default="./res/", type=str,
                    help="path to directory for store results")
    ap.add_argument("-c", "--classes", dest="classes", default="./class_names.txt", type=str,
                    help="file with class labels")
    args = vars(ap.parse_args())
    # print(args)

    print("Start...")

    # get JSON files
    files = sorted(glob.glob(os.path.join(args['input'], '*.json')))
    print("LabelMe files:", files)

    # read classes from file
    with open(args['classes']) as fs:
        classes = fs.read().splitlines()
    print("class_names:", classes)

    # create dirs
    dst_images_dir = os.path.join(args['output'], "images")
    dst_labels_dir = os.path.join(args['output'], "labels")
    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)
    if not os.path.exists(dst_labels_dir):
        os.makedirs(dst_labels_dir)

    for filename in files:
        print(filename)
        with open(filename, 'r') as fs:
            dataj = json.load(fs)

            image_name = dataj["imagePath"]
            print("Read", image_name)
            image_path = os.path.join(args['input'], image_name)
            image = cv2.imread(image_path)
            if image is None:
                print("Error read image by imagePath:", image_path)
                # try by name
                image_name = os.path.splitext(
                    os.path.basename(filename))[0] + ".jpg"
                image_path = os.path.join(args['input'], image_name)
                print("Try read image by name:", image_path)
                image = cv2.imread(image_path)
                if image is None:
                    print("Error read image by name:", image_path)
                    if dataj["imageData"]:
                        print("Get image data from imageData")
                        image_b64_data = dataj["imageData"]
                        image_bin_data = base64.b64decode(image_b64_data)
                        image = cv2.imdecode(np.asarray(
                            bytearray(image_bin_data), dtype=np.uint8), flags=cv2.IMREAD_COLOR)
                        print("Decoded image:", image.shape)
                        print("Save image", image_path)
                        cv2.imwrite(image_path, image)

            # print(image.shape)
            height, width = image.shape[:2]
            print("Image size:", width, height)

            # copy image
            dst_image_path = os.path.join(dst_images_dir, image_name)
            print("Copy image to:", dst_image_path)
            shutil.copyfile(image_path, dst_image_path)

            shapes = dataj["shapes"]

            res_file_name = os.path.splitext(
                os.path.basename(filename))[0] + ".txt"
            print(res_file_name)

            dst_label_path = os.path.join(dst_labels_dir, res_file_name)
            try:
                fw = open(dst_label_path, "a+")
            except:
                print("Error open file!")

            for item in shapes:
                item_class = item["label"]
                class_id = classes.index(item_class)
                point1, point2 = item["points"][0], item["points"][1]
                print(point1, point2)

                x1, y1 = point1
                x2, y2 = point2
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

                print("MM:", xmin, xmax, ymin, ymax)
                box = (xmin, ymin, xmax, ymax)

                #box = point1[0], point1[1], point2[0], point2[1]
                print(item_class, class_id, box)
                bb = convert((width, height), box)
                # class_idx x_center y_center width height
                out_str = F"{class_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n"
                print(out_str)
                fw.write(out_str)
            fw.close()
    print("Done.")


if __name__ == '__main__':
    main()
