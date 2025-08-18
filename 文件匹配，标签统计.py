import shutil
from pathlib import Path


def check_and_clean_dataset(dataset_type="val"):
    """
    检查指定数据集（train/val/test）中图片与标签的对应关系，删除不匹配的文件，并统计每类标签的框数.

    Args:
        dataset_type: 数据集类型，可选 'train', 'val', 'test'
    """
    # 配置信息（从data.yaml提取）
    root_path = Path("D:/训练数据/yolodataset")
    names = ["fire", "head", "helmet", "person", "smoke"]
    nc = len(names)  # 类别数量

    # 图片和标签文件夹路径
    img_dir = root_path / f"images/{dataset_type}"
    label_dir = root_path / f"labels/{dataset_type}"

    # 确保文件夹存在
    if not img_dir.exists():
        print(f"图片文件夹不存在: {img_dir}")
        return
    if not label_dir.exists():
        print(f"标签文件夹不存在: {label_dir}")
        return

    # 创建不匹配文件的存放目录
    mismatch_dir = root_path / f"mismatched_files/{dataset_type}"
    mismatch_img_dir = mismatch_dir / "images"
    mismatch_label_dir = mismatch_dir / "labels"
    mismatch_img_dir.mkdir(parents=True, exist_ok=True)
    mismatch_label_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片和标签的文件名（不含扩展名）
    img_files = set()
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        img_files.update([p.stem for p in img_dir.glob(f"*{ext}")])

    label_files = set(p.stem for p in label_dir.glob("*.txt"))

    # 找出不匹配的文件
    img_only = img_files - label_files  # 只有图片没有标签
    label_only = label_files - img_files  # 只有标签没有图片

    # 移动不匹配的图片
    for img_stem in img_only:
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            img_path = img_dir / f"{img_stem}{ext}"
            if img_path.exists():
                shutil.move(str(img_path), str(mismatch_img_dir))
                print(f"移动无标签图片: {img_path.name} -> 不匹配文件夹")
                break

    # 移动不匹配的标签
    for label_stem in label_only:
        label_path = label_dir / f"{label_stem}.txt"
        if label_path.exists():
            shutil.move(str(label_path), str(mismatch_label_dir))
            print(f"移动无图片标签: {label_path.name} -> 不匹配文件夹")

    # 统计每类标签的框数
    class_counts = {name: 0 for name in names}
    total_boxes = 0

    # 遍历所有匹配的标签文件
    for label_stem in img_files & label_files:
        label_path = label_dir / f"{label_stem}.txt"
        if not label_path.exists():
            continue

        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:  # 至少包含类别和4个坐标
                    continue
                try:
                    class_idx = int(parts[0])
                    if 0 <= class_idx < nc:
                        class_name = names[class_idx]
                        class_counts[class_name] += 1
                        total_boxes += 1
                except ValueError:
                    continue  # 忽略格式错误的行

    # 输出统计结果
    print("\n" + "=" * 50)
    print(f"数据集 {dataset_type} 处理完成")
    print(f"不匹配的图片数量: {len(img_only)}")
    print(f"不匹配的标签数量: {len(label_only)}")
    print(f"总有效标签框数量: {total_boxes}")
    print("\n各类别标签框数量统计:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 检查val集（可改为'train'或'test'）
    check_and_clean_dataset(dataset_type="val")
    # 如需同时检查多个数据集，可以取消下面的注释
    # check_and_clean_dataset(dataset_type='train')
    # check_and_clean_dataset(dataset_type='test')
