"""
独自yoloモデルを使って、動画に対して物体検出を行う

対応動画フォーマットの例（OpenCV依存）
MP4 (.mp4)
AVI (.avi)
MOV (.mov)
MKV (.mkv)
WebM (.webm)
FLV (.flv)
WMV (.wmv)

q 終了
s フレームをスキップ
1,2,3 画像サイズ変更
"""
from ultralytics import YOLO
import cv2
#from picamera2 import Picamera2
from imutils.video import FPS
import time 

print()
print("qキーの入力で終了します。")
time.sleep(1)

# YOLOのモデルを読み込み
model = YOLO('runs/detect/train/weights/best.pt')

model_name = model.ckpt_path # モデルファイルのパス
print("yoloモデル:",model_name)  

# 動画ファイルを開く
video_path = "myMovie.mp4"

cap = cv2.VideoCapture(video_path)
print("movie_file:",video_path)  
print()

window_name = model_name + " Movie"
# ウィンドウを作成（WINDOW_NORMALでリサイズ可能にする）
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# 任意の大きさにリサイズ
cv2.resizeWindow(window_name, 640, 480)  # 幅640、高さ480

# FPS計測開始
fps = FPS().start()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 動画終了

    # キー入力待ち（qで終了）
    key = cv2.waitKey(30) & 0xFF  # 下位8ビットを取得
    if key == ord("q"):
        # print('**********q')
        break
    elif key == ord("s"):
        # フレームをスキップ
        print('**********Skip')
        for _ in range(300):
            cap.read()
    elif key == ord("1"):
        cv2.resizeWindow(window_name, 320, 240)  # 小さく
    elif key == ord("2"):
        cv2.resizeWindow(window_name, 640, 480)  # 普通サイズ
    elif key == ord("3"):
        cv2.resizeWindow(window_name, 960, 720) # 大きく
    else:
        # print('*-------')
        # YOLOで推論（BGR画像そのままでOK）
        # 進行状況バー（tqdm）表示
        results = model(frame)

        # 進行状況バー（tqdm）非表示
        # 人間だけ検出する 検出するクラスを指定する
        # ただし、modelはそのままなので、スピードは変わらない
        # results = model(frame, classes=[0], verbose=False)

        # 検出された画像を取得（OpenCV形式のnumpy配列）
        annotated_frame = results[0].plot()
        # 表示 ウィンドウのタイトル
        cv2.imshow(window_name, annotated_frame)

        # yoloが見つけたクラスの数をターミナルに表示
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy()     # クラスID
        confidences = boxes.conf.cpu().numpy()  # 信頼度

        # 条件: クラスID==0(person) かつ 信頼度 >= 0.6
        # mask = (class_ids == 0) & (confidences >= 0.6)
        mask = (confidences >= 0.6)
        person_count = mask.sum()
        #print("人間",person_count)  
    fps.update()

# FPS計測終了
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# 終了処理
cap.release()
cv2.destroyAllWindows()
