#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'code'))
	print(os.getcwd())
except:
	pass

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import pytesseract
import cv2
import os
import glob
import matplotlib.pyplot as plt

os.chdir('..')
os.chdir('Images')
f = open('./list.txt', 'r')
categories = f.read().split('\n')[:-1]
categories.sort()


#%%
for i, category in enumerate(categories):
    print(category)
    for name in glob.glob('./' + category + '/*'):
        img = cv2.imread(name)
        
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b]) 
        
        plt.imshow(img)
        plt.show()
        print(pytesseract.image_to_string(img, lang='Hangul'))


#%%



