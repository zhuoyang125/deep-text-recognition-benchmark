import cv2
import random
import argparse
import numpy as np
import seaborn as sns

image = cv2.imread('landscape.jpg')
height = image.shape[0]
width = image.shape[1]
print(height)
print(width)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', help='num of images', required=True)
args = parser.parse_args()

color_palette = np.array(sns.color_palette('hls',8))*255

#crops a 100 x 100 image
def rand_crop(image):
    random.seed(a=None)
    rand_height = random.randint(0,height-101)
    rand_width = random.randint(0,width-101)
    cropped = image[rand_height:rand_height+100, rand_width:rand_width+100]
    return cropped

def rand_num(image,display_info):
    random.seed(a=None)
    #trandom coordinates and font scale
    scale = 0.65
    font_choice = [0,2,3,4,6,7]
    font = random.choice(font_choice)
    color = color_palette[random.randint(0,71) % 8]

    
    cv2.putText(image, str(display_info), 
                (10,50), font, 
                scale, color, 1)
    return image

def rand_float():
    base = random.random()
    scaled_value = 360*base
    rounded_value = round(scaled_value,3)
    return rounded_value

try:
   val = int(args.num)
except ValueError:
   print('Argument not integer')

#f = open('gt_test.txt','w+')
i = 0
for _ in range(val):
    output = image.copy()
    random.seed(a=None)
    rounded_value = rand_float()
    cropped = rand_crop(output)
    numbered = rand_num(cropped,rounded_value)
    cv2.imwrite('test' + str(i) +'.jpg',numbered)
    #f.write('converted_custom_data_validation/Float_Numbers/image' + str(i) + '.jpg\t' + str(rounded_value) + '\n')
    i += 1
#f.close()


    
    

    