import os
import shutil

# 源数据集路径
source_root = r"D:/训练数据/yolodataset"
source_val_images = os.path.join(source_root, "images", "val")
source_val_labels = os.path.join(source_root, "labels", "val")

# 目标路径
target_root = r"D:\yang\Downloads\烟,火数据集\data\val"
target_images = os.path.join(target_root, "images")
target_labels = os.path.join(target_root, "labels")

# 确保目标文件夹存在
os.makedirs(target_images, exist_ok=True)
os.makedirs(target_labels, exist_ok=True)

# person对应的类别ID (根据提供的names: 3: person)
PERSON_CLASS_ID = 3

# 统计变量
total_files = 0
moved_files = 0

# 获取所有标签文件
label_files = [f for f in os.listdir(source_val_labels) if f.endswith(".txt")]

print(f"发现 {len(label_files)} 个标签文件，开始处理...\n")

for label_file in label_files:
    total_files += 1
    label_path = os.path.join(source_val_labels, label_file)

    # 对应的图像文件路径
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_file = None
    for ext in image_extensions:
        img_path = os.path.join(source_val_images, os.path.splitext(label_file)[0] + ext)
        if os.path.exists(img_path):
            image_file = img_path
            break

    if not image_file:
        print(f"警告: 未找到 {label_file} 对应的图像文件，已跳过")
        continue

    # 检查标签文件是否包含person类别
    contains_person = False
    with open(label_path, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts:  # 确保行不为空
                class_id = int(parts[0])
                if class_id == PERSON_CLASS_ID:
                    contains_person = True
                    break  # 找到后立即退出循环

    # 如果包含person类别，则移动文件
    if contains_person:
        # 移动标签文件
        target_label_path = os.path.join(target_labels, label_file)
        # 检查目标文件是否已存在，如果存在则跳过
        if os.path.exists(target_label_path):
            print(f"文件 {label_file} 已存在于目标位置，已跳过")
            continue

        shutil.move(label_path, target_label_path)

        # 移动图像文件
        target_image_path = os.path.join(target_images, os.path.basename(image_file))
        shutil.move(image_file, target_image_path)

        moved_files += 1
        print(f"已移动 {label_file} 及其图像 (包含person类别)")

# 显示处理结果
print("\n处理完成!")
print(f"总处理文件数: {total_files}")
print(f"包含person并移动的文件数: {moved_files}")
print(f"目标图像文件夹: {target_images}")
print(f"目标标签文件夹: {target_labels}")
