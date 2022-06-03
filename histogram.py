import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("./images/fiber2.png", 0)
cv2.imshow("Act",img)
cv2.waitKey(0)

def histogram_equalization(img):
    equ = cv2.equalizeHist(img)
    cv2.imshow("eq",equ)
    cv2.waitKey(0)
    return equ

histogram_equalization(img)
# cv2.imshow("chekc",img)
# row = len(img)
# col = len(img[0])

# def plot(pixel, count):
#   plt.bar(pixel, count)
#   plt.show

# count = np.zeros(256, dtype = int)
# pixel = np.zeros(256, dtype = int)
# for i in img:
#   for j in i:
#     pixel[j] = j
#     count[j] += 1
# h = np.zeros(256, dtype = float)
# for i in range(256):
#   h[i] = count[i]/(row*col)
# H = np.zeros(256, dtype = float)
# H[0] = h[0]
# for i in range(1, 256):
#   H[i] = h[i]+H[i-1]
# Fin = np.zeros([img.shape[0], img.shape[1]], dtype = float)
# for i in range(row):
#   for j in range(col):
#     pix = img[i, j]
#     S = -1
#     S = 255*H[pix]
#     Fin[i, j] = S

# count1 = np.zeros(256, dtype = int)
# pixel1 = np.zeros(256, dtype = int)
# for i in Fin:
#   for j in i:
#     j = (int)(j)
#     pixel1[j] = j
#     count1[j] += 1
# plt.imshow(Fin, cmap = 'gray')
# cv2.imshow("Image",Fin)
# cv2.waitKey(0)