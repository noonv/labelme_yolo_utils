# labelme_yolo_utils
Utils for [labelme](https://github.com/wkentaro/labelme) and [YOLOv5](https://github.com/ultralytics/yolov5) detector

## convert_labelme2yolo
Convert LabelMe JSON to YOLO txt.

example:
```
./convert_labelme2yolo.py --input=./photos --output=./res --classes=./class_names.txt
```

## convert_yolo2labelme
Convert YOLO txt labels to LabelMe JSON.

example:
```
./convert_yolo2labelme.py --input=./res --output=./res_json --classes=./res/class_names.txt
```

Takes data from input path: read images from "images" and labels from "labels" directories.

Example:
```
    ./input_dir
    |-> /images/*.jpg
    `-> /labels/*.txt
```

## predetect_yolo2labelme
Make detection on image and store results into LabelMe JSON format  for next manual labeling.
Could be useful for processes of Semi-supervised learning or Active Learning.

example:
```
./predetect_yolo2labelme.py --input=./photos/ --model=./yolov5s.pt --classes=./coco_class_names.txt --threshold=0.3
```

## copy_unlabeled_images
Copy (or move) images without LabelMe JSON files to output directory.

example:
```
# copy
copy_unlabeled_images.py --input=./images --output=./unlabeled --extention="jpg"
# move 
copy_unlabeled_images.py --input=./images --output=./unlabeled --extention="jpg" --move
```
## cut_image_from_labelme
Cut image from LabelMe JSON bounding box to image file.

example:
```
./cut_image_from_labelme.py --input=./images --output=./bboxes
```
