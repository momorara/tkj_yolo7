# -*- coding: utf-8 -*-
# -----------------------------------------
# ランダム画像を選んで推論し、
# スペースで次の画像、qで終了
# -----------------------------------------
"""
license
GNU Affero General Public License v3（AGPL v3）
"""
import os
import random
import matplotlib.pyplot as plt
from glob import glob
from ultralytics import YOLO
import shutil

# # 学習済みモデルの読み込み
# model = load_model('cats_vs_dogs_cnn.h5')

# 1. 学習済みモデルのパス
# YOLOv8の学習結果は通常、'runs/detect/train' または 'runs/detect/trainX' に保存されます。
# ここでは、学習が完了した際の最も性能が良い重みファイル (best.pt) を指定します。
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# 推論対象フォルダ（猫・犬の両方を含む上位フォルダ）
base_dir = 'dataset_tv/images/val'


img_list = []

folder_path = os.path.join(base_dir)
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
img_list.extend(files)

if not img_list:
    print("画像ファイルが見つかりません。フォルダのパスを確認してください。")
    exit()

# -----------------------------------------
# matplotlib ウィンドウを作成
# -----------------------------------------
fig, ax = plt.subplots()
plt.axis('off')  # 軸を非表示

# グローバル変数で表示用画像
img_show = None

# 1. モデルのロード
# 学習済みモデルの重みファイルを指定してYOLOオブジェクトを作成します。
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ モデルを正常にロードしました: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ エラー: モデルファイルが見つかりません。パスを確認してください: {MODEL_PATH}")
    exit(0)

# 3. 結果の保存先ディレクトリ名
PROJECT_NAME = 'custom_inference' 

def show_random_image(event=None):
    """ランダムに画像を選んで推論＆表示"""
    global img_show
    img_path = random.choice(img_list)

    # predictメソッドを使用して推論を実行
    results = model.predict(
        source=img_path,  # 推論対象
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



    jpg_files = glob(os.path.join(results[0].save_dir, "*.jpg"))

    # Matplotlibで結果画像を表示
    img = plt.imread(jpg_files[0])
    plt.imshow(img)
    plt.axis("off")
    plt.title("YOLO 7 Category")
    plt.show()
    plt.pause(0.1)  # 少し待つ

def on_key(event):
    if event.key == ' ':
        # 新しい画像を読む前にフォルダを削除
        try:
            shutil.rmtree(PROJECT_NAME)
            # print(f"削除しました: {save_dir}")
        except Exception as e:
            print(f"ディレクトリ削除失敗: {e}")
        show_random_image()
    elif event.key.lower() == 'q':
        plt.close(fig)
        plt.ioff()  # 終了時にオフ

# キーイベントを接続
fig.canvas.mpl_connect('key_press_event', on_key)

# 最初の画像を表示
show_random_image()

plt.show()
