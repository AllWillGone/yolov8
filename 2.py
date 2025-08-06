import os
import shutil
from collections import Counter

# 配置路径
base_path = "D:/训练数据/yolodataset"
test_img_dir = os.path.join(base_path, "images/test")
test_label_dir = os.path.join(base_path, "labels/test")  # 假设标签目录与图像目录平行
val_img_dir = os.path.join(base_path, "images/val")
val_label_dir = os.path.join(base_path, "labels/val")

# 确保目标目录存在
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 类别映射
class_names = ["fire", "head", "helmet", "person", "smoke"]


# 移动包含fire的测试集文件
def move_fire_files():
    moved_count = 0
    # 遍历所有测试集标签文件
    for label_file in os.listdir(test_label_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(test_label_dir, label_file)
        img_file = os.path.splitext(label_file)[0] + ".jpg"  # 假设图片为jpg格式

        # 检查图片是否存在
        img_path = os.path.join(test_img_dir, img_file)
        if not os.path.exists(img_path):
            # 尝试其他常见图片格式
            for ext in ['.png', '.jpeg', '.bmp']:
                alt_img_path = os.path.join(test_img_dir, os.path.splitext(label_file)[0] + ext)
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
                    img_file = os.path.splitext(label_file)[0] + ext
                    break
            else:
                print(f"警告: 找不到图片文件 {img_file} 或替代格式")
                continue

        # 检查标签是否包含fire（类别0）
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == '0':  # fire类别
                    # 移动标签文件
                    shutil.move(label_path, os.path.join(val_label_dir, label_file))

                    # 移动图片文件
                    shutil.move(img_path, os.path.join(val_img_dir, img_file))

                    moved_count += 1
                    break  # 找到一个fire即移动，跳出当前文件循环
    return moved_count


# 统计验证集各类别数量
def count_val_objects():
    counter = Counter()
    for label_file in os.listdir(val_label_dir):
        if not label_file.endswith(".txt"):
            continue

        with open(os.path.join(val_label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id < len(class_names):
                        counter[class_id] += 1
    return counter


# 执行移动操作
print("开始移动包含fire的测试集文件到验证集...")
moved_count = move_fire_files()
print(f"\n移动完成! 共移动了 {moved_count} 个包含fire的文件到验证集")

# 统计最终val集各类别数量
val_counts = count_val_objects()
print("\n验证集各类别统计:")
for class_id, count in val_counts.items():
    print(f"{class_names[class_id]}: {count} 个标注框")

# 打印总框数和fire类别数量
print(f"\n验证集总标注框数量: {sum(val_counts.values())}")
print(f"验证集中 'fire' 类别数量: {val_counts.get(0, 0)}")
sudo mkdir -p /etc/docker && sudo tee /etc/docker/daemon.json <<-'EOF'
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.imgdb.de",
        "https://docker-0.unsee.tech",
        "https://docker.hlmirror.com"
    ]
}
EOF
sudo systemctl daemon-reload && sudo systemctl restart docker
