import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import os.path
import cv2

from sklearn.cluster import KMeans

max_image = 6
max_cluster = 2 ** 6

def create_out_path():
  path = './out'
  if(os.path.isdir(path)):
    shutil.rmtree(path)

  os.makedirs(path)
  count = 1
  while(count <= max_image):
    os.makedirs(f'./out/{count}')
    count = count + 1

def clustering(image):
  print(f'clustering image: {image_count}')

  reshaped_array = image.reshape(-1, 3)
  cv2.imwrite(f'./out/{image_count}/{image_count}_ORIGINAL.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  cluster = 3
  while(cluster <= max_cluster):
    kmeans = KMeans(n_clusters=cluster, n_init=10).fit(reshaped_array)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)
    cv2.imwrite(f'./out/{image_count}/{image_count}_{cluster}.png', cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    cluster = cluster * 2

def main():
  global image_count
  create_out_path()

  image_count = 1
  while(image_count <= max_image):
    image_path = f'./img/{image_count}.jpg'
    if(os.path.isfile(image_path)):
      clustering(mpl.image.imread(image_path))
    image_count = image_count + 1

main()
