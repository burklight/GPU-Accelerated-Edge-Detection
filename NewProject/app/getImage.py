import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img_path = input('Which is the path of the image you want to detect the edges of? ')

img = mpimg.imread(img_path)
gray = 255*rgb2gray(img)
gray = gray.astype(int)

np.savetxt(fname='tmp_img.txt', X = gray, delimiter=',', newline=',', fmt='%d')

shape_file  = open("tmp_input.txt", "w")
shape_file.write(str(gray.shape[0]) + " " + str(gray.shape[1])+ " " + 'tmp_img.txt')
shape_file.close()
