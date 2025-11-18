# -*- coding: utf-8 -*-
"""
yolo detect train data=data.yaml model=yolov8n.pt epochs=3 imgsz=320
として処理した結果を使う

対象画像は.  SOURCE_PATで指定
"""
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from glob import glob

# --- 設定 ---

# 1. 学習済みモデルのパス
# YOLOv8の学習結果は通常、'runs/detect/train' または 'runs/detect/trainX' に保存されます。
# ここでは、学習が完了した際の最も性能が良い重みファイル (best.pt) を指定します。
MODEL_PATH = 'runs/detect/train/weights/best.pt'
# MODEL_PATH = '/Volumes/Lexar/yolo/runs/detect/train/weights/last.pt'

# 2. 検出対象の画像またはフォルダのパス
# 例: 'my_test_image.jpg' または 'path/to/new_images_folder'
SOURCE_PATH = 'test3.png'
# SOURCE_PATH = '/Volumes/Lexar/yolo7/catdog.png'

# 3. 結果の保存先ディレクトリ名
PROJECT_NAME = 'custom_inference' 


# --- 推論実行 ---
def run_inference():
    """YOLOv8モデルを使って推論を実行する関数"""
    
    # 1. モデルのロード
    # 学習済みモデルの重みファイルを指定してYOLOオブジェクトを作成します。
    try:
        model = YOLO(MODEL_PATH)
        print(f"モデルを正常にロードしました: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません。パスを確認してください: {MODEL_PATH}")
        return

    # 2. 検出の実行
    print(f"推論を {SOURCE_PATH} に対して実行中...")
    
    # predictメソッドを使用して推論を実行
    results = model.predict(
        source=SOURCE_PATH,  # 推論対象
        conf=0.25,           # 信頼度閾値 (デフォルト: 0.25)
        iou=0.7,             # IOU閾値 (重複バウンディングボックスの除去用)
        save=True,           # 検出結果の画像保存を有効化
        project=PROJECT_NAME # 結果を保存するルートディレクトリ名
    )
    # 結果の取得
    for result in results:
        boxes = result.boxes  # バウンディングボックス情報
        for box in boxes:
            # 座標情報（x1, y1, x2, y2）: 左上と右下
            x1, y1, x2, y2 = box.xyxy[0]  # tensor型 → floatに変換
            conf = box.conf[0]            # 信頼度 (0〜1)
            cls = int(box.cls[0])         # クラス番号
            label = result.names[cls]     # クラス名（例: person, dog）

            # # バウンディングボックスの描画
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(conf,cls,label,x1, y1, x2, y2)

    # 3. 結果の出力場所
    # 結果は通常、プロジェクト名と連番のフォルダに保存されます (例: custom_inference/predict/)
    # 正確な出力パスは実行後にコンソールに表示されます。
    
    print("推論が完了しました。")
    print(f"結果の画像は '{PROJECT_NAME}/predict' のようなフォルダに保存されています。")

    jpg_files = glob(os.path.join(results[0].save_dir, "*.jpg"))
    # Matplotlibで結果画像を表示
    img = plt.imread(jpg_files[0])
    plt.imshow(img)
    plt.axis("off")
    plt.title("YOLO 7 Category")
    plt.show()

if __name__ == "__main__":
    run_inference()