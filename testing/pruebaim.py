from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# Open the two images
img_png = Image.open("this.png").convert("RGB")
img_jpg = Image.open("mimi.jpg").convert("RGB")


# Convert to numpy arrays
arr_png = np.array(img_png)
arr_jpg = np.array(img_jpg)

# Print sizes
print("PNG image size (width, height):", img_png.size)
print("PNG image shape (height, width, channels):", arr_png.shape)

print("\nJPG image size (width, height):", img_jpg.size)
print("JPG image shape (height, width, channels):", arr_jpg.shape)

# Print full matrices (be careful: large images produce lots of output!)
print("\nPNG image matrix:\n", arr_png)
print("\nJPG image matrix:\n", arr_jpg)

# Display the two images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_png)
plt.title("PNG Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_jpg)
plt.title("JPG Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Create an output directory if it doesn’t exist
os.makedirs("output_images", exist_ok=True)

# Save both images (you can change the format if you want)
img_png.save("output_images/image_from.png", quality=95)  # save PNG as JPG
img_jpg.save("output_images/image_from.jpg")              # save JPG as PNG

print("✅ Images saved in the folder 'output_images'")