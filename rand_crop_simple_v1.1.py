import cv2
import random
import argparse
import numpy as np
import seaborn as sns
import skimage
import skimage.io
import matplotlib.pyplot as plt
from random import randint

#adds noise to image
def plotnoise(img):
    mode = ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"]
    noised = skimage.util.random_noise(img, mode=random.choice(mode))
    return noised

#crops a 100 x 100 image
def rand_crop(image, label_width, label_height):
    random.seed(a=None)
    padding = 10
    rand_height = random.randint(padding,height-label_height)
    rand_width = random.randint(padding,width-label_width)
    cropped = image[rand_height - padding:rand_height + label_height + padding, rand_width - padding:rand_width + label_width + padding]
    return cropped

def get_slice(image,formatted_value):
    random.seed(a=None)
    #trandom coordinates and font scale
    # if (upper_dp-1) > 3:
    #     scale = random.uniform(0.30,0.50)
    # else:
    #     scale = random.uniform(0.55,0.70)
    scale = random.uniform(0.5, 8)
    font_choice = [0,2,3,4,6,7]
    font = random.choice(font_choice)
    #text_org = (10,10)
    #color = color_palette[random.randint(0,71) % 8]
    (label_width, label_height), baseline = (cv2.getTextSize(str(formatted_value),font,scale,1))
    sliced = rand_crop(image, label_width, label_height)
    text_org = (10, sliced.shape[0] - 10)

    '''
    text = '{}:{:0.2f}%'.format(det['category_id]', det['score']*100)
    '''
    cv2.putText(sliced, str(formatted_value), 
                text_org, font, 
                scale, (0,0,0), 1)
    #(label_width, label_height), baseline = (cv2.getTextSize(str(formatted_value),font,scale,1))
    #print(label_width, label_height)
    #slice_x = text_org[0]
    #slice_y = text_org[1] - label_height
    #padding = 8
    #sliced = image[slice_y - padding : slice_y + label_height + padding, slice_x - padding : slice_x + label_width + padding]
    return sliced


def rand_float():
    dp = random.choice(range(lower_dp,upper_dp))
    base = random.random()
    scaled_value = 360*base
    rounded_value = (round(scaled_value,dp))
    formatted_value = format(rounded_value,'.{}f'.format(dp))
    return formatted_value


if __name__ == '__main__':

    image = cv2.imread('Sky.jpg')
    image = image/255
    height = image.shape[0]
    width = image.shape[1]
    print("Original image's height:" + str(height))
    print("Original image's width:" + str(width))

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', help='num of images', required=True)
    parser.add_argument("-d", "--decimalpt", nargs=2, action="store", help="specify range of dp",required=True)
    args = parser.parse_args()

    try:
        val = int(args.num)
    except ValueError:
        print('Argument is not an integer')

    #color_palette = np.array(sns.color_palette('hls',8))*255
    lower_dp = int(args.decimalpt[0])
    upper_dp = int(args.decimalpt[1]) + 1 #add 1 to include upper dp
    
    f = open('gt_validation.txt','w+')
    #i = 0
    for i in range(val):
        output = image.copy()
        random.seed(a=None)
        formatted_value = rand_float()
        #cropped = rand_crop(output)
        #numbered = get_slice(cropped,formatted_value)
        numbered = get_slice(output,formatted_value)

        value = randint(1, 10)
        if value >= 4:
            numbered = plotnoise(numbered)
        
        #Preview Image
        #cv2.imshow('Numbered', numbered)
        #cv2.waitKey(0)

        #Write Image to File
        numbered = cv2.convertScaleAbs(numbered, alpha=(255.0))
        cv2.imwrite('validation' + str(i) +'.jpg', numbered)

        f.write('validation/validation' + str(i) + '.jpg\t' + str(formatted_value) + '\n')
        #i += 1
    f.close()


    
    

    