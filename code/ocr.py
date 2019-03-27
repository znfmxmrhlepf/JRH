import pytesseract
import cv2
import os
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')
os.chdir('Images')
f = open('./list.txt', 'r')
categories = f.readlines()
categories.sort()

for i, category in enumerate(categories):
    print(str(i) + ' : ' + category, end='')

imgList = glob.glob('./' + categories[int(input())][:-1] + '/*')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
for name in imgList:
    img = cv2.imread(name)
    print(pytesseract.image_to_string(img, lang='Hangul'))
    cv2.resize(img , (img.shape[0] // 2, img.shape[1] // 2))
    cv2.imshow('img', img)
    cv2.waitKey(0)