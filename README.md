# labelme_yolo_utils
utils for [labelme](https://github.com/wkentaro/labelme) and [YOLOv5](https://github.com/ultralytics/yolov5) detector

## convert_labelme2yolo
convert LabelMe JSON to YOLO txt

example:
```
./convert_labelme2yolo.py --input=./photos --output=./res --classes=./class_names.txt
```

## predetect_yolo2labelme
make detection on image and store results into LabelMe JSON format  for next manual labeling

example:
```
./predetect_yolo2labelme.py --input=./photos/ --model=./yolov5s.pt --classes=./coco_class_names.txt --threshold=0.3
```
