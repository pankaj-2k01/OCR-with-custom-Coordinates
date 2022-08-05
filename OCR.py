from cgitb import text
import cv2
from pytesseract import *
import numpy as np
import seaborn as sns


class OCR:
  def __init__(self,image_path,x,y,w,h) :
    self.image_path=image_path
    self.x=x
    self.y=y
    self.w=w
    self.h=h

  def show_valueimage(self,flag,img):
    cv2.imshow(flag,img)
    cv2.waitKey(0)

  def preprocess_img(self,img):
    gray=self.grayscale(img)
    self.show_valueimage("Grayscal Image",gray)
    
    # hist_equalize=self.histogram_equalization(gray)
    # self.show_valueimage("Histogram Equalizes",hist_equalize)
    # rem_noise=self.remove_noise(gray)
    # self.show_valueimage("Noise",rem_noise)
    blur = self.blur(gray)
    self.show_valueimage("Blurred Image",blur)
    threshold_image=self.thresholding(blur)
    self.show_valueimage("Threshold Image",threshold_image)
    erroded_image=self.erossion(threshold_image)
    self.show_valueimage("Erroded Image",erroded_image)
    diluted_image=self.dilution(erroded_image)
    self.show_valueimage("Dilated Image",diluted_image)
    
    return diluted_image
    
  def histogram_equalization(self,img):
    equ = cv2.equalizeHist(img)
    return equ

  def grayscale(self,img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  def blur(self,img):
    # return cv2.GaussianBlur(img,(5,5),0)
    return img

  def thresholding(self,img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  
  def InvOTSUthresholding(self,img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  
  def distancetransform(self,img):
    img=cv2.distanceTransform(img,cv2.DIST_L2,5)
    img=(img*255).astype("uint8")
    return self.InvOTSUthresholding(img)
  
  def MorphologicalEx(self,img):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
  
  def erossion(self,img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(img, kernel, iterations=1)

  def dilution(self,img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

  def canny(self,img):
    return cv2.Canny(img, 100, 200)

  def deskew(self,img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

  def remove_noise(self,img):
    return cv2.medianBlur(img,5)

  def read_image(self):
    img=cv2.imread(self.image_path)
    img=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    return img

  def cropped_img(self,img2):
    cropped = img2[self.y:self.y + self.h, self.x:self.x + self.w]
    return cropped

  def get_text(self,img):
    text = pytesseract.image_to_string(img)
    return text
  
  def print_text(self,text):
    print(text)

if __name__=="__main__":
  # ocr=OCR("./images/test4.png",25,30,300,130)
  ocr=OCR("./images/fiber2.png",350,550,300,150)
  
  img=ocr.read_image() 
  cropped_image=ocr.cropped_img(img)
  preprocess=ocr.preprocess_img(cropped_image)
  textualdata=ocr.get_text(preprocess)
  ocr.print_text(textualdata)



# ocr=OCR("./images/fiber3.png",250,350,250,75) 
# ocr=OCR("./images/test6.png",0,0,134,137)
# ocr=OCR("./images/test8.png",100,100,500,400) 
# ocr=OCR("./images/fiber3.png",250,350,250,75) 
# ocr=OCR("./images/fiber1.png",300,400,500,300)
# ocr=OCR("./images/test5.png",210,170,130,140)






