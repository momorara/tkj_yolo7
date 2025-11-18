# -*- coding: utf-8 -*-
"""
license
GNU Affero General Public License v3（AGPL v3）

YOLO多クラス分類 推論サンプル（全画像順次処理）
- Ultralytics YOLOv8 学習済みモデルを使用
- valフォルダ内に全クラスの画像が混在
- ファイル名の先頭にクラス名が含まれている形式に対応
"""

import os
import shutil
from ultralytics import YOLO

# ==============================
# 設定
# ==============================
MODEL_PATH = 'runs/detect/train/weights/best.pt'
VAL_DIR = 'dataset_tv/images/val'
CLASSES = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']
ERR_DIR = 'result_err'
ERR_SAVE = True  # 誤判定画像を保存するか

# ==============================
# 前処理
# ==============================
# result_errを削除して新規作成
if os.path.exists(ERR_DIR):
    shutil.rmtree(ERR_DIR)
os.makedirs(ERR_DIR, exist_ok=True)

# YOLOモデルの読み込み
model = YOLO(MODEL_PATH)

# 結果格納用
stats = {cls: {"total": 0, "correct": 0, "wrong": 0} for cls in CLASSES}

# ==============================
# メインループ
# ==============================
print("推論を開始します...\n")

for fname in os.listdir(VAL_DIR):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    if "._" in fname:
        continue

    img_path = os.path.join(VAL_DIR, fname)

    # ファイル名から正解クラスを特定
    fname_lower = fname.lower()
    true_cls = None
    for cls in CLASSES:
        if fname_lower.startswith(cls.lower()):
            true_cls = cls
            break

    if true_cls is None:
        print(f"⚠️ クラス名を判定できません: {fname}")
        continue

    stats[true_cls]["total"] += 1

    try:
        # 推論実行
        results = model.predict(img_path, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # 最も信頼度が高い予測を使用
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            max_idx = confs.argmax()
            pred_cls_name = model.names[cls_ids[max_idx]].lower()
        else:
            pred_cls_name = "none"

        # 正解・誤判定チェック
        if pred_cls_name == true_cls.lower():
            stats[true_cls]["correct"] += 1
        else:
            stats[true_cls]["wrong"] += 1
            if ERR_SAVE:
                err_subdir = os.path.join(ERR_DIR, true_cls, pred_cls_name)
                os.makedirs(err_subdir, exist_ok=True)
                shutil.copy(img_path, err_subdir)

    except Exception as e:
        print(f"⚠️ エラー: {img_path} -> {e}")

print("\n推論完了\n")

# ==============================
# 結果表示
# ==============================
print("====== 結果 ======")
total_all = correct_all = 0
for cls in CLASSES:
    total = stats[cls]["total"]
    correct = stats[cls]["correct"]
    wrong = stats[cls]["wrong"]
    acc = (correct / total * 100) if total > 0 else 0
    print(f"{cls:10s}: {total:4d}枚  正解={correct:4d}  誤判定={wrong:4d}  正解率={acc:5.1f}%")
    total_all += total
    correct_all += correct

if total_all > 0:
    overall_acc = correct_all / total_all * 100
    print(f"\n総合正解率: {overall_acc:.1f}% ({correct_all}/{total_all})")
else:
    print("画像が見つかりませんでした。")
