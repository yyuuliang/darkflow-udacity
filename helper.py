import os
import sys
import csv
import cv2


# copy file to sample folder according to csv file

csv_fname = os.path.join('/home/notus/Github/yyuuliang/darkflow/udacity/udacity-one.csv')
sourcefolder ='/home/notus/Github/yyuuliang/darkflow/udacity/object-dataset/'
outfolder ='/home/notus/Github/yyuuliang/darkflow/sample_img'

total = 0
with open(csv_fname, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )
    for row in spamreader:
        total = total +1
        img_name = row[0]
        source_path = os.path.join(sourcefolder, img_name)   
        # print(source_path)
        imgcv = cv2.imread(source_path)
        out_path = os.path.join(outfolder, img_name)
        # print(out_path)
        cv2.imwrite(out_path, imgcv)

print('copied {0} images to {1}'.format(total,outfolder))