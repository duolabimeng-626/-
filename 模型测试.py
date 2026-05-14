import os
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# ===================== 【Windows 必加！】 =====================
if __name__ == '__main__':

    # ===================== 你的路径 直接用 =====================
    TEST_IMAGES_DIR = r"C:\Users\86183\Desktop\dataset\images\test"
    TEST_LABELS_DIR = r"C:\Users\86183\Desktop\dataset\labels\test"

    MODEL_A_PATH = r"C:\Users\86183\Desktop\深度学习项目\small_object_20260504_130330\weights\best.pt"
    MODEL_B_PATH = r"C:\Users\86183\Desktop\深度学习项目\small_object_20260505_005739\weights\best.pt"

    SAVE_ROOT = r"C:\Users\86183\Desktop\模型评估结果"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # ===================== GPU 加速配置 =====================
    BATCH_SIZE = 8
    CONF_THRESH = 0.3
    IOU_THRESH = 0.5
    MAX_SAVE_IMAGES = 100

    CLASS_NAMES = ["small_object"]

    # ===================== 加载模型 =====================
    model_a = YOLO(MODEL_A_PATH)
    model_b = YOLO(MODEL_B_PATH)

    # ===================== 评估函数 =====================
    def evaluate_model(model, model_name, save_dir):
        print(f"\n===== {model_name} 开始 GPU 评估 =====")
        os.makedirs(save_dir, exist_ok=True)
        vis_dir = os.path.join(save_dir, "检测效果图")
        os.makedirs(vis_dir, exist_ok=True)

        img_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('jpg','png','jpeg'))]
        saved_count = 0

        for img_name in img_files:
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            results = model(img_path, conf=CONF_THRESH, iou=IOU_THRESH, batch=BATCH_SIZE, verbose=False)

            if saved_count < MAX_SAVE_IMAGES:
                vis_img = results[0].plot()
                cv2.imwrite(os.path.join(vis_dir, img_name), vis_img)
                saved_count += 1

        # 官方评估
        metrics = model.val(
            data=r"C:\Users\86183\Desktop\dataset\data.yaml",
            imgsz=640,
            batch=BATCH_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            device=0,
            save_dir=os.path.join(save_dir, "官方指标"),
            workers=0  # 关键：防止多进程报错
        )

        result = {
            "模型": model_name,
            "精确率(P)": round(metrics.box.mp, 3),
            "召回率(R)": round(metrics.box.mr, 3),
            "mAP50": round(metrics.box.map50, 3),
            "mAP50-95": round(metrics.box.map, 3)
        }

        with open(os.path.join(save_dir, "评估报告.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"{model_name} 评估完成！")
        return result

    # ===================== 开始运行 =====================
    res_a = evaluate_model(model_a, "模型A", os.path.join(SAVE_ROOT, "模型A"))
    res_b = evaluate_model(model_b, "模型B", os.path.join(SAVE_ROOT, "模型B"))

    # ===================== 双模型对比图 =====================
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(figsize=(10, 6))

    names = ["模型A", "模型B"]
    ps = [res_a["精确率(P)"], res_b["精确率(P)"]]
    rs = [res_a["召回率(R)"], res_b["召回率(R)"]]
    aps = [res_a["mAP50"], res_b["mAP50"]]

    x = [0,1]
    w = 0.25
    plt.bar([i-w for i in x], ps, w, label="精确率")
    plt.bar(x, rs, w, label="召回率")
    plt.bar([i+w for i in x], aps, w, label="mAP50")

    plt.title("双模型测试集指标对比")
    plt.xticks(x, names)
    plt.legend()
    plt.savefig(os.path.join(SAVE_ROOT, "双模型对比图.png"), dpi=300)
    plt.close()

    print("\n✅ 全部评估完成！结果保存在：", SAVE_ROOT)