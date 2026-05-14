import os
import pandas as pd
import shutil

# ===================== 【只需修改这 2 个路径】 =====================
TRAIN_CSV = r"C:\Users\邓天航\PycharmProjects\PythonProject2\test\train\train\_annotations.csv"
VALID_CSV = r"C:\Users\邓天航\PycharmProjects\PythonProject2\test\valid\valid\_annotations.csv"

# 输出根目录（自动创建，不用管）
OUTPUT_ROOT = r"C:\Users\邓天航\PycharmProjects\PythonProject2\dataset"
# ==================================================================

# 类别映射
CLASS_MAP = {
    "Hardhat": 0,
    "NO-Hardhat": 1
}

def process_set(csv_path, mode):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 {mode} CSV：{csv_path}")
        return

    # 原图片目录
    img_src_dir = os.path.dirname(csv_path)
    # 目标目录
    img_dst_dir = os.path.join(OUTPUT_ROOT, "images", mode)
    lbl_dst_dir = os.path.join(OUTPUT_ROOT, "labels", mode)
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    # 读取标注
    df = pd.read_csv(csv_path)
    grouped = df.groupby("filename")

    idx = 1
    for filename, group in grouped:
        src_img = os.path.join(img_src_dir, filename)
        if not os.path.exists(src_img):
            continue

        # ===================== 重命名图片 =====================
        dst_img = os.path.join(img_dst_dir, f"{idx}.jpg")
        shutil.copy(src_img, dst_img)

        # ===================== 生成 YOLO 标签 =====================
        img_w = group["width"].iloc[0]
        img_h = group["height"].iloc[0]
        lines = []

        for _, row in group.iterrows():
            cls = CLASS_MAP[row["class"]]
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

            # YOLO 格式计算
            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 保存标签
        with open(os.path.join(lbl_dst_dir, f"{idx}.txt"), "w") as f:
            f.write("\n".join(lines))

        print(f"[{mode}] {filename} → {idx}.jpg")
        idx += 1

if __name__ == "__main__":
    print("=== 开始处理训练集 train ===")
    process_set(TRAIN_CSV, "train")

    print("\n=== 开始处理验证集 valid ===")
    process_set(VALID_CSV, "valid")

    print("\n🎉 全部完成！数据集已生成在：\n", OUTPUT_ROOT)