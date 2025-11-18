# -*- coding: utf-8 -*-
"""
license
GNU Affero General Public License v3（AGPL v3）

yolo detect train data=data.yaml model=yolov8n.pt epochs=3 imgsz=320
として処理した結果を使う

対象画像はdata.yamlで指定
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="data.yaml", epochs=40, imgsz=128)
