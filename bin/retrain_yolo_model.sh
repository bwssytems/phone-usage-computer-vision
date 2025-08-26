#!/bin/bash
mkdir workspace
cd workspace

# Use GPU 0 for training
#yolo detect train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=16 device=0

# Use CPU for training
yolo detect train model=yolo11n.pt data=../data.yaml epochs=100 imgsz=640 batch=16