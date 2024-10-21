import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("resim.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (1024, 768))
# Histogram
histogramx, bin = np.histogram(img.flatten(), 256, [0, 256])
cdf = histogramx.cumsum()
cdf_normal = cdf * histogramx.max() / cdf.max()
plt.plot(cdf_normal, color="b")
plt.hist(img.flatten(), 256, [0, 256], color="r")
plt.xlim([0, 256])
plt.legend(("cdf", "histogram"), loc="upper left")
# histogram eşitleme uygula
esit = cv2.equalizeHist(img)
cv2.imshow("histogram resmi", esit)
cv2.waitKey(0)
cv2.destroyAllWindows()
