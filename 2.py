import os
import shutil
from collections import defaultdict

# 配置信息
config = {
    "path": "D:/训练数据/yolodataset",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 5,
    "names": ["fire", "head", "helmet", "person", "smoke"],
}

# 定义路径
dataset_path = config["path"]
val_image_dir = os.path.join(dataset_path, config["val"])
test_image_dir = os.path.join(dataset_path, config["test"])
val_label_dir = os.path.join(dataset_path, config["val"].replace("images", "labels"))
test_label_dir = os.path.join(dataset_path, config["test"].replace("images", "labels"))

# 统计类别数量
names = config["names"]
class_counts = {split: defaultdict(int) for split in ["val", "test"]}


# 帮助函数：获取标签文件中的类别统计
def get_class_stats(label_path):
    class_stats = defaultdict(int)
    with open(label_path) as lf:
        for line in lf.readlines():
            class_id = line.strip().split(" ")[0]
            if class_id.isdigit() and 0 <= int(class_id) < len(names):
                class_stats[names[int(class_id)]] += 1
    return class_stats


# 步骤1：从验证集删除只包含helmet的图片，最多减少3000个框
def remove_helmet_only_from_val():
    removed_count = 0
    val_images = os.listdir(val_image_dir)

    # 遍历验证集图片
    for img_file in val_images:
        img_path = os.path.join(val_image_dir, img_file)
        label_file = img_file.split(".")[0] + ".txt"
        label_path = os.path.join(val_label_dir, label_file)

        # 检查标签是否存在
        if not os.path.exists(label_path):
            continue

        # 获取标签中的类别统计
        class_stats = get_class_stats(label_path)

        # 如果标签中只有helmet
        if set(class_stats.keys()) == {"helmet"}:
            # 删除图片和标签
            os.remove(img_path)
            os.remove(label_path)
            removed_count += class_stats["helmet"]
            print(f"Deleted from val: {img_file} and {label_file}")

            # 检查是否达到最大删除框数
            if removed_count >= 3000:
                break

    print(f"Removed {removed_count} boxes from val set.")


# 步骤2：从测试集优先移入只含有fire的图片到验证集
def move_fire_only_from_test_to_val():
    moved_count = 0

    # 遍历测试集图片
    test_images = os.listdir(test_image_dir)
    for img_file in test_images:
        img_path = os.path.join(test_image_dir, img_file)
        label_file = img_file.split(".")[0] + ".txt"
        label_path = os.path.join(test_label_dir, label_file)

        # 检查标签是否存在
        if not os.path.exists(label_path):
            continue

        # 获取标签中的类别统计
        class_stats = get_class_stats(label_path)

        # 如果标签中只有fire
        if set(class_stats.keys()) == {"fire"}:
            # 移动图片和标签到验证集
            shutil.move(img_path, os.path.join(val_image_dir, img_file))
            shutil.move(label_path, os.path.join(val_label_dir, label_file))
            moved_count += class_stats["fire"]
            print(f"Moved to val: {img_file} and {label_file}")

    print(f"Moved {moved_count} fire boxes to val set.")


# 步骤3：从测试集移入fire多于smoke的图片到验证集
def move_fire_more_than_smoke_from_test_to_val():
    moved_count = 0

    # 遍历测试集图片
    test_images = os.listdir(test_image_dir)
    for img_file in test_images:
        img_path = os.path.join(test_image_dir, img_file)
        label_file = img_file.split(".")[0] + ".txt"
        label_path = os.path.join(test_label_dir, label_file)

        # 检查标签是否存在
        if not os.path.exists(label_path):
            continue

        # 获取标签中的类别统计
        class_stats = get_class_stats(label_path)

        # 如果标签中有fire和smoke，且fire数量多于smoke
        if "fire" in class_stats and "smoke" in class_stats and class_stats["fire"] > class_stats["smoke"]:
            # 移动图片和标签到验证集
            shutil.move(img_path, os.path.join(val_image_dir, img_file))
            shutil.move(label_path, os.path.join(val_label_dir, label_file))
            moved_count += sum(class_stats.values())
            print(f"Moved to val: {img_file} and {label_file}")

    print(f"Moved {moved_count} boxes (fire>smoke) to val set.")


# 执行所有步骤
print("Step 1: Removing only-helmet images from validation set...")
remove_helmet_only_from_val()

print("\nStep 2: Moving fire-only images from test to validation set...")
move_fire_only_from_test_to_val()

print("\nStep 3: Moving fire-more-than-smoke images from test to validation set...")
move_fire_more_than_smoke_from_test_to_val()

# 重新统计类别数量
# 获取所有图片文件名（不包含扩展名）
val_images = os.listdir(val_image_dir)
val_image_files = [f.split(".")[0] for f in val_images]
test_images = os.listdir(test_image_dir)
test_image_files = [f.split(".")[0] for f in test_images]

# 统计验证集
for img_file in val_image_files:
    label_file = img_file + ".txt"
    label_path = os.path.join(val_label_dir, label_file)
    if os.path.exists(label_path):
        with open(label_path) as lf:
            for line in lf.readlines():
                class_id = line.strip().split(" ")[0]
                if class_id.isdigit() and 0 <= int(class_id) < len(names):
                    class_counts["val"][names[int(class_id)]] += 1

# 统计测试集
for img_file in test_image_files:
    label_file = img_file + ".txt"
    label_path = os.path.join(test_label_dir, label_file)
    if os.path.exists(label_path):
        with open(label_path) as lf:
            for line in lf.readlines():
                class_id = line.strip().split(" ")[0]
                if class_id.isdigit() and 0 <= int(class_id) < len(names):
                    class_counts["test"][names[int(class_id)]] += 1

# 输出统计结果
print("\nFinal Statistics:")

for split in ["val", "test"]:
    print(f"\n{split} set:")
    total_boxes = sum(class_counts[split].values())
    print(f"Total boxes: {total_boxes}")
    for class_name in names:
        print(f"  {class_name}: {class_counts[split][class_name]}")
