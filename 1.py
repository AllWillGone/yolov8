import os
from PIL import Image

# 根路径
root = r"D:\训练数据\yolodataset\images"
sub_dirs = ["train"]

for sub in sub_dirs:
    img_dir = os.path.join(root, sub)
    bad_dir = os.path.join(img_dir, "_bad")  # 存放无法修复的图片
    os.makedirs(bad_dir, exist_ok=True)

    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # 跳过非图片文件

        try:
            img = Image.open(img_path).convert("RGB")
            base, ext = os.path.splitext(filename)
            new_path = os.path.join(img_dir, base + ".jpg")
            if ext.lower() != ".jpg":
                img.save(new_path, "JPEG", quality=95)
                os.remove(img_path)  # 删除原文件
                print(f"[修复] {sub}/{filename} -> {base}.jpg")
            else:
                img.save(img_path, "JPEG", quality=95)  # 原地覆盖
                print(f"[覆盖] {sub}/{filename}")
        except Exception as e:
            shutil.move(img_path, os.path.join(bad_dir, filename))
            print(f"[损坏] {sub}/{filename} 已移到 _bad")

print("✅ 图片清洗完成，请检查 _bad 文件夹，确认无误后可删除。")