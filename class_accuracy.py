import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

# Config
base_dir = './samples'  # Root directory containing Class0, Class1, etc.
show_grid = True  # Set to False if you only want counts

# Track counts
class_counts = defaultdict(int)

# Get all class folders
class_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("Class")]

# Count images per class
for class_name in sorted(class_folders):
    class_path = os.path.join(base_dir, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
    class_counts[class_name] = len(image_files)

# Print results
print("📊 Image Counts per Class:")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls}: {count} images")

# Optional: display samples
if show_grid:
    for cls in sorted(class_counts):
        print(f"\n🖼 Showing samples from {cls}:")
        class_dir = os.path.join(base_dir, cls)
        img_paths = sorted(os.listdir(class_dir))[:9]  # Show first 9 images
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        fig.suptitle(f'{cls} Samples', fontsize=14)
        for ax, img_file in zip(axes.flatten(), img_paths):
            img = Image.open(os.path.join(class_dir, img_file))
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
