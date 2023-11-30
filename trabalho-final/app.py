import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

clusters = 10

image = mpl.image.imread("./img/1.png")

plt.imshow(image)

image.shape

x = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=clusters, n_init=10)

kmeans.fit(x)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

plt.imshow(segmented_img / 255)

import cv2

cv2.imwrite(f"./img/out/1_ORIGINAL.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite(f"./img/out/1_{clusters}.png", cv2.cvtColor(segmented_img.astype("uint8"), cv2.COLOR_BGR2RGB))