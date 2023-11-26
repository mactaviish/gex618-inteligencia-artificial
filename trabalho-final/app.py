import os
import numpy as np
import cv2 as cv

def imagem_existe(imagem):
  return os.path.exists(imagem)

def tudo():
  algoritmo = cv.CascadeClassifier('haarcascade_smile.xml')
  count = 1

  while count <= 1:
    img_dir = f"img/{count}.png"

    if imagem_existe(img_dir):
      print(f"Encontrou a imagem {img_dir}")
      imagem = cv.imread(img_dir)
      sorrisos = algoritmo.detectMultiScale(imagem)
      cv.imshow("Sorrisos", sorrisos)
    else:
      print(f"'{img_dir}' não existe!")

    count = count + 1

def main():
  tudo()
  cv.waitKey()

main()