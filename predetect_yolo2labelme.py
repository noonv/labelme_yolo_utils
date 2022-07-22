#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
make detection on image and store results into LabelMe JSON format 
for next manual labeling

example:
./predetect_yolo2labelme.py --input=./photos/ --model=./yolov5s.pt --classes=./coco_class_names.txt --threshold=0.3
./predetect_yolo2labelme.py --input=./photos --model=../yolo_test/runs/train/exp27/weights/best.pt --classes=./class_names.txt --threshold=0.3

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

#import labelme
#import yolov5

try:
    import labelme
except ModuleNotFoundError:
    is_labelme_available = False

try:
    import yolov5
    import torch
except ModuleNotFoundError:
    is_yolov5_available = False

device = "cuda:0"  # or "cpu"


def check_module_available(module_name):
    module_spec = importlib.util.find_spec(module_name)
    found = module_spec is not None
    return found


def load_model(model_path, device="0"):
    # init yolov5 model

    is_cuda = device.isnumeric()
    if is_cuda:
        #device = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
        device = "cuda:"+device if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            print("CUDA device name:", torch.cuda.get_device_name())
    else:
        #device = torch.device("cpu")
        device = "cpu"
    print("device:", device)

    yolo = yolov5.YOLOv5(model_path, device)
    return yolo


def model_predict(model, filename, size):
    # predict
    results = model.predict(filename, size=size)
    # print(results)
    # print(results.pandas().xyxy[0])

    # parse results
    predictions = results.pred[0]

    #boxes = predictions[:, :4]
    #scores = predictions[:, 4]
    #categories = predictions[:, 5]
    #print(boxes, scores, categories)

    return predictions


def convert_predictions_to_shapes(predictions, class_names, threshold):
    # convert yolo predictions to labelme shapes
    shapes = []

    pr = predictions.cpu().numpy()
    # n,6 -> x1, y1, x2, y2, score, category
    #print(type(pr), pr.shape, pr)

    if pr.size == 0:
        print("No any predictions!")
        return []

    for row in pr:
        print("prediction:", row)
        x1, y1 = float(row[0]), float(row[1])
        x2, y2 = float(row[2]), float(row[3])
        points = [(x1, y1), (x2, y2)]
        score = row[4]
        label = class_names[int(row[5])]
        shape = dict(
            label=label,
            points=points,  # [(p[0], p[1]) for p in points],
            group_id=None,
            shape_type="rectangle",
            flags={},
        )
        if score >= threshold:
            print(shape)
            shapes.append(shape)
        else:
            print("Low score for shape:", score)

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
        # base64.b64encode(cv2.imread(image_filename)).decode("utf-8")  # base64.b64encode(image).decode("utf-8")
        dataj["imageData"] = None
        dataj["imageHeight"] = imageHeight
        dataj["imageWidth"] = imageWidth

        # print(dataj)

        try:
            # save JSON data into file
            with open(filename, 'w') as fs:
                json.dump(dataj, fs, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Error write file:", filename)
            raise e

    return 0


def main():
    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="input", default="./photos", type=str,
                    help="path to directory with images")
    ap.add_argument("-c", "--classes", dest="classes", default="./coco_class_names.txt", type=str,
                    help="file with class labels")
    ap.add_argument("-m", "--model", dest="model", default="./yolov5s.pt", type=str,
                    help="model fot detection")
    ap.add_argument("-t", "--threshold", dest="threshold", default=0.3, type=float,
                    help="threshold for model detection")
    ap.add_argument("-e", "--extention", dest="extention", default="jpg", type=str,
                    help="extention of image files")
    ap.add_argument("-d", "--device", dest="device", default="0",
                    type=str, help="device for model (cuda - 0) or cpu")
    args = vars(ap.parse_args())
    print(args)

    print("Start...")

    # check extention
    ext = args['extention']
    if ext.startswith('*'):
        ext = ext[1:]
    if ext.startswith('.'):
        ext = ext[1:]
    print("Image extention:", ext)

    # check - is labelme available?
    is_labelme_available = False
    if check_module_available("labelme"):
        #import labelme
        is_labelme_available = True
        print("labelme found!")
    else:
        print("labelme not found!")

    # check - is yolov5 available?
    is_yolov5_available = False
    if check_module_available("yolov5"):
        #import yolov5
        is_yolov5_available = True
        print("yolov5 found!")
    else:
        print("yolov5 not found!")

    # load yolo model
    if is_yolov5_available:
        model_path = args['model']
        model = load_model(model_path, device=args["device"])

    # get image files
    files = sorted(glob.glob(os.path.join(args['input'], "*."+ext)))
    print("Images:", files)

    # read classes from file
    with open(args['classes']) as fs:
        classes = fs.read().splitlines()
    print("class_names:", classes)

    for filename in files:
        print("Read", filename)

        res_file_name = os.path.splitext(
            os.path.basename(filename))[0] + ".json"
        dst_file_path = os.path.join(args['input'], res_file_name)

        # read image and get it size
        image = cv2.imread(filename)
        if image is None:
            print("Error read image:", filename)

        height, width = image.shape[:2]
        print("Image size:", width, height)

        shapes = []

        # detect objects here
        predictions = None
        if is_yolov5_available:
            predictions = model_predict(
                model, filename, size=max(width, height))

            shapes = convert_predictions_to_shapes(
                predictions, class_names=classes, threshold=args['threshold'])

        # print(shapes)

        if len(shapes) == 0:
            print("No shapes for save!")
            continue

        # save to JSON-file
        imagePath = os.path.basename(filename)

        save_data_to_json(is_labelme_available=is_labelme_available,
                          image_filename=filename,
                          filename=dst_file_path,
                          shapes=shapes,
                          imagePath=imagePath,
                          imageHeight=height,
                          imageWidth=width)

    print("Done.")


if __name__ == '__main__':
    main()
