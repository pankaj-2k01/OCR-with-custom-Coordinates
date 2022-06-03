import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_image(image_path):
  img=cv2.imread(image_path)
  greyscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret,threshold=cv2.threshold(greyscale,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(18,18))
  dilation=cv2.dilate(threshold,kernel)
  contours, hierarchy=cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

  img2=img.copy()
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = img2[y:y + h, x:x + w]
    text = pytesseract.image_to_string(cropped)
    print(text)

def ocr(image_path,x,y,w,h):
  img=cv2.imread(image_path)
  greyscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret,threshold=cv2.threshold(greyscale,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(18,18))
  dilation=cv2.dilate(threshold,kernel)
  contours, hierarchy=cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  img2=img.copy()

  rect = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
  cropped = img2[y:y + h, x:x + w]
  text = pytesseract.image_to_string(cropped)
  print(text)

# ocr("test.png",5,137,147,164)
# get_image("test.png")

class OCR:
  def __init__(self,image_path,x,y,w,h) :
    self.image_path=image_path
    self.x=x
    self.y=y
    self.w=w
    self.h=h

  def preprocess_img(self):
    gray=cv2.imread(self.image_path,0)
    img2=gray.copy()
    # img2=cv2.imread(self.image_path)
    # kernel = np.ones((2, 1), np.uint8)
    ret,threshold=cv2.threshold(img2,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(18,18))
    img = cv2.erode(img2, kernel, iterations=1)
    img = cv2.dilate(img, threshold,kernel, iterations=1)
    return img2
    
  
  def cropped_img(self,img2):
    cropped = img2[self.y:self.y + self.h, self.x:self.x + self.w]
    # cv2.imshow("image",cropped)
    # cv2.waitKey(100000)
    return cropped

  def get_text(self,cropped_img):
    text = pytesseract.image_to_string(cropped_img)
    return text
  
  def print_text(self,text):
    print(text)
  
# ocr=OCR("test6.png",0,0,134,137)
# ocr=OCR("test5.png",210,170,130,140)
# ocr=OCR("test4.png",25,30,300,130)
# ocr=OCR("test8.png",100,100,500,400) turned out image 
preprocess_image=ocr.preprocess_img()
cropped_image=ocr.cropped_img(preprocess_image)
textual_data=ocr.get_text(cropped_image)
ocr.print_text(textual_data)


