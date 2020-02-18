import cv2
import random
import argparse
import numpy as np
import seaborn as sns

image = cv2.imread('sky.jpg')
height = image.shape[0]
width = image.shape[1]
print(height)
print(width)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', help='num of images', required=True)
parser.add_argument("-d", "--decimalpt", nargs=2, action="store", help="specify range of dp",required=True)
args = parser.parse_args()

#color_palette = np.array(sns.color_palette('hls',8))*255
lower_dp = int(args.decimalpt[0])
upper_dp = int(args.decimalpt[1]) + 1 #add 1 to include upper dp


#crops a 100 x 100 image
def rand_crop(image):
    random.seed(a=None)
    rand_height = random.randint(0,height-101)
    rand_width = random.randint(0,width-101)
    cropped = image[rand_height:rand_height+100, rand_width:rand_width+100]
    return cropped

def get_slice(image,formatted_value):
    random.seed(a=None)
    #trandom coordinates and font scale
    if (upper_dp-1) > 3:
        scale = random.uniform(0.30,0.50)
    else:
        scale = random.uniform(0.55,0.70)
    font_choice = [0,2,3,4,6,7]
    font = random.choice(font_choice)
    text_org = (10,50)
    #color = color_palette[random.randint(0,71) % 8]

    '''
    text = '{}:{:0.2f}%'.format(det['category_id]', det['score']*100)
    '''
    cv2.putText(image, str(formatted_value), 
                text_org, font, 
                scale, (0,0,0), 1)
    (label_width, label_height), baseline = (cv2.getTextSize(str(formatted_value),font,scale,1))
    slice_x = text_org[0]
    slice_y = text_org[1] - label_height
    padding = 8
    sliced = image[slice_y - padding : slice_y + label_height + padding, slice_x - padding : slice_x + label_width + padding]
    return sliced


def rand_float():
    dp = random.choice(range(lower_dp,upper_dp))
    base = random.random()
    scaled_value = 360*base
    rounded_value = (round(scaled_value,dp))
    formatted_value = format(rounded_value,'.{}f'.format(dp))
    return formatted_value

try:
   val = int(args.num)
except ValueError:
   print('Argument is not an integer')

#f = open('gt_test.txt','w+')
i = 0
for _ in range(val):
    output = image.copy()
    random.seed(a=None)
    formatted_value = rand_float()
    cropped = rand_crop(output)
    numbered = get_slice(cropped,formatted_value)
    
    #Preview Image
    cv2.imshow('Numbered', numbered)
    cv2.waitKey(0)

    #Write Image to File
    #cv2.imwrite('test' + str(i) +'.jpg',numbered)
    #f.write('converted_custom_data_validation/Float_Numbers/image' + str(i) + '.jpg\t' + str(formatted_value) + '\n')
    i += 1
#f.close()


    
    

    