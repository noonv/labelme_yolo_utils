#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cut image from LabelMe JSON bounding box to image file

1. create dirs output, output/images
2. create file_name_rect?.jpg in output directory.

example:
./cut_image_from_labelme.py --input=./fp --output=./fp/res

"""

__author__ = "Vladimir"

import os
import json
import cv2
import glob
import argparse
import shutil
import base64
import numpy as np


def main():

    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="input", default="./images", type=str,
                    help="path to directory with images and LabelMe JSON files")
    ap.add_argument("-o", "--output", required=True, dest="output", default="./res", type=str,
                    help="path to directory for store results")
    ap.add_argument("-e", "--extention", dest="extention", default="jpg", type=str,
                    help="extention of image files")
    args = vars(ap.parse_args())
    # print(args)

    print("Start...")

    # check extention
    ext = args['extention']
    if ext.startswith('*'):
        ext = ext[1:]
    if ext.startswith('.'):
        ext = ext[1:]
    print("Image extention:", ext)

    # get JSON files
    files = sorted(glob.glob(os.path.join(args['input'], '*.json')))
    print("LabelMe files:", files)

    # create dirs
    dst_images_dir = args['output']

    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)

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
                    else:
                        print("Error get image data!")
                        break

            # print(image.shape)
            height, width = image.shape[:2]
            print("Image size:", width, height)

            shapes = dataj["shapes"]

            shape_counter = 0
            for item in shapes:
                item_class = item["label"]
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
                print(shape_counter, item_class, box)

                res_file_name = os.path.splitext(
                    os.path.basename(filename))[0] + "_rect"+str(shape_counter) + "." + ext
                print(res_file_name)

                dst_image_path = os.path.join(dst_images_dir, res_file_name)

                def chkv(val):
                    if val < 0:
                        return 0
                    return int(val)

                # cut shape rect from image
                shape_image = image[chkv(ymin):chkv(
                    ymax), chkv(xmin):chkv(xmax)]
                # save rect to file
                res = cv2.imwrite(dst_image_path, shape_image)
                if not res:
                    print("Error save image:", dst_image_path)

                shape_counter += 1

    print("Done.")


if __name__ == '__main__':
    main()
