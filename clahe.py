# CLAHE
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("resim.jpg", 0)
claheTek = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# CLAHE işlemi
clahe_image = claheTek.apply(img)
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title("Orjinal resim")
plt.imshow(img, cmap="gray")
plt.subplot(222)
plt.title("Histogram - Orjinal resim")
plt.hist(img.ravel(), bins=256, range=[0, 256], color="black", alpha=0.7)
plt.subplot(223)
plt.title("CLAHE işlenmiş Image")
plt.imshow(clahe_image, cmap="gray")
plt.subplot(224)
plt.title("Histogram - CLAHE Uygulaması")
plt.hist(clahe_image.ravel(), bins=256, range=[0, 256], color="black", alpha=0.7)
plt.show()
