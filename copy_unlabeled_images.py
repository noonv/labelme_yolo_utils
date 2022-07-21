#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
copy images without LabelMe JSON files to output directory

1. create output directory
2. copy images without json file to output directory

example:
copy_unlabeled_images.py --input=./images --output=./unlabeled --extention="jpg"

'''

__author__ = 'Vladimir'

import os
import glob
import argparse
import shutil


def main():

    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="input", default="images", type=str,
                    help="path to directory with images and LabelMe JSON files")
    ap.add_argument("-o", "--output", dest="output", default="unlabeled", type=str,
                    help="path to directory for store images without labels")
    ap.add_argument("-e", "--extention", dest="extention", default="jpg", type=str,
                    help="extention of image files")
    ap.add_argument("-m", "--move", nargs='?', dest="move", const=True, default=False, help='move images instead of copy')
    args = vars(ap.parse_args())
    # print(args)

    print("Start...")

    # get JSON files
    json_files = sorted(glob.glob(os.path.join(args['input'], '*.json')))
    print("LabelMe files:", json_files)

    # check extention
    ext = args['extention']
    if ext.startswith('*'):
        ext = ext[1:]
    if ext.startswith('.'):
        ext = ext[1:]
    print("Image extention:", ext)

    is_move = args["move"]

    # get image files
    image_files = sorted(glob.glob(os.path.join(
        args['input'], "*." + ext)))
    print("Image files:", image_files)

    # create list of image filenames with LabelMe JSON files
    images_with_labels = []
    for json_filename in json_files:
        base_name = os.path.splitext(os.path.basename(json_filename))[0]
        image_filename_with_json = base_name + "." + ext
        image_path = os.path.join(args['input'], image_filename_with_json)
        images_with_labels.append(image_path)
    print("Images with labels:", images_with_labels)

    # create dirs
    dst_images_dir = args['output']
    os.makedirs(dst_images_dir, exist_ok=True)

    print("Process images...")
    for image_filename in image_files:
        print(image_filename)
        if image_filename in images_with_labels:
            print("Has label.")
        else:
            print("No label.")

            dst_image_path = os.path.join(
                dst_images_dir, os.path.basename(image_filename))
            if is_move:
                # move image
                print("Move image to:", dst_image_path)
                shutil.move(image_filename, dst_image_path)
            else:
                # copy image
                print("Copy image to:", dst_image_path)
                shutil.copyfile(image_filename, dst_image_path)
    print("Done.")


if __name__ == '__main__':
    main()
