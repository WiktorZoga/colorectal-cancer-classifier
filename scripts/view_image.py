import sys
from PIL import Image
import matplotlib.pyplot as plt

image_path = sys.argv[1]

try:
    img = Image.open(image_path)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"File: {image_path}\nSize: {img.size}")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"ERROR: {e}")