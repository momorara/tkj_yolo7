# -*- coding: utf-8 -*-
"""
画像から簡易的にラベルを作ります。

本来はしっかりアノテーションして作るべきものですが、
対象画像がほぼアノテーション結果で切り取られているものとして処理します。

python -m pip install pillow

dataset_tvを対象にラベルをつくり　同じディレクトリにlabelsフォルダを作ります
"""
import os
from PIL import Image
import shutil

# --- 設定 ---
SOURCE_DIR = "dataset_tv"  # 既存の画像データセットのルートディレクトリ名
TARGET_DIR = "dataset_l" # 新しいYOLO形式のデータセットのルートディレクトリ名

# クラス定義 (ファイル名のプレフィックスに基づきクラスIDを決定)
CLASSES = {
    "bike": 0,
    "cars": 1,
    "cats": 2,
    "dogs": 3,
    "flowers": 4,
    "horses": 5,
    "human": 6
}

# バウンディングボックスの縮小率 (0.8 = 画像の幅・高さの80%を使用)
# 画像全体がオブジェクトであると仮定し、上下左右それぞれ5%ずつ内側に縮小する
SCALE_FACTOR = 0.8

# --- ディレクトリ構造の定義 ---
SPLITS = ['train', 'val']

def create_target_structure():
    """新しいYOLOv8形式のディレクトリ構造を作成する"""
    print(f"ターゲットディレクトリ '{TARGET_DIR}' を作成中...")
    
    # imagesフォルダとlabelsフォルダを作成
    for folder in ['images', 'labels']:
        for split in SPLITS:
            os.makedirs(os.path.join(TARGET_DIR, folder, split), exist_ok=True)
            
    print("ディレクトリ構造の作成が完了しました。")

def create_yolo_labels_and_copy_images():
    """画像をコピーし、ラベルファイルを生成する"""
    for split in SPLITS:
        source_image_dir = os.path.join(SOURCE_DIR, 'images', split)
        
        target_image_dir = os.path.join(TARGET_DIR, 'images', split)
        target_label_dir = os.path.join(TARGET_DIR, 'labels', split)

        if not os.path.exists(source_image_dir):
            print(f"⚠️ 警告: ソース画像ディレクトリ {source_image_dir} が見つかりません。スキップします。")
            continue

        print(f"\n--- {split.upper()} データの処理を開始 ---")
        
        processed_count = 0
        deleted_count = 0

        for filename in os.listdir(source_image_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            source_image_path = os.path.join(source_image_dir, filename)
            target_image_path = os.path.join(target_image_dir, filename)
            
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            target_label_path = os.path.join(target_label_dir, label_filename)

            # 1. クラスIDの決定
            class_name_prefix = filename.split('_')[0].lower()
            if class_name_prefix not in CLASSES:
                print(f"⚠️ 警告: {filename} のプレフィックスにクラスIDが未定義です。スキップします。")
                continue

            class_id = CLASSES[class_name_prefix]

            # 2. 画像の読み込みとエラー処理 (読み込めないファイルを検出)
            try:
                img = Image.open(source_image_path)
                width, height = img.size
                
                # 3. 読み込み成功した場合、新しいディレクトリに画像をコピー
                shutil.copy2(source_image_path, target_image_path)
                
            except Exception as e:
                # Pillowが画像を認識できない、または破損している場合
                print(f"🛑 エラー: '{source_image_path}' を読み込めません ('{e}')。ファイルをスキップ・削除します。")
                
                # 破損ファイルは元のディレクトリから削除
                try:
                    os.remove(source_image_path)
                    deleted_count += 1
                except:
                    print(f"❌ 元ファイルの削除に失敗しました: {source_image_path}")
                continue

            # 4. YOLO形式座標の計算 (縮小したバウンディングボックス)
            # 座標は画像の中心 (0.5, 0.5) で固定
            x_center_norm = 0.5
            y_center_norm = 0.5
            
            # 幅と高さをSCALE_FACTORで縮小
            w_norm = SCALE_FACTOR 
            h_norm = SCALE_FACTOR

            # 5. ラベルファイルの書き出し
            with open(target_label_path, 'w') as f:
                # YOLO形式: [class_id] [x_center] [y_center] [width] [height]
                f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            processed_count += 1

        print(f"  ✅ {split.upper()}処理完了: {processed_count} 個の画像とラベルを生成。{deleted_count} 個の破損ファイルを削除しました。")


# --- メイン処理 ---
if __name__ == "__main__":
    create_target_structure()
    create_yolo_labels_and_copy_images()
    print("\n🎉 データセットの変換が完了しました。")
    print(f"新しいYOLO形式のデータセットは '{TARGET_DIR}' に作成されました。")