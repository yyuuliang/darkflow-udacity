"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import csv
import cv2

def udacity_voc_csv(ANN, pick, exclusive = False, test = False):

    # print('Parsing for {} {}'.format(
    #     pick, 'exclusively' * int(exclusive)))
    def pp(l): # pretty printing 
        for i in l: print('{}: {}'.format(i,l[i]))
    
    def parse(line): # exclude the xml tag
        x = line.decode().split('>')[1].decode().split('<')[0]
        try: r = int(x)
        except: r = x
        return r

    def _int(literal): # for literals supposed to be int 
        return int(float(literal))
    
    dumps = list()

    # csv_fname = os.path.join('/home/yan/data/udacity_data/udacity.csv')
    csv_fname = os.path.join('/home/notus/whitebase/darkflow-udacity/Autti/overfit-train.csv')
    if test:
        print('load validation csv')
        csv_fname = os.path.join('/home/notus/whitebase/darkflow-udacity/Autti/overfit-train.csv')

    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )

        # crowdai dataset
        # for row in spamreader:
        #     row = row[0].split(',')
        #     img_name = row[4]
        #     w = 1920
        #     h = 1200
      
        #     # labels = row[1:]
        #     all = list()
        #     # for i in range(0, len(labels), 5):
        #     xmin = int(row[0])
        #     ymin = int(row[1])
        #     xmax = int(row[2])
        #     ymax = int(row[3])
        #     class_name = row[5]
        #     all += [[class_name, xmin, ymin, xmax, ymax]]

        #     add = [[img_name, [w, h, all]]]
        #     dumps += add

        # Autti dataset
        for row in spamreader:
            # row = row[0].split(',')
            img_name = row[0]
            w = 1920
            h = 1200
      
            # labels = row[1:]
            all = list()
            # for i in range(0, len(labels), 5):
            xmin = int(row[1])
            ymin = int(row[2])
            xmax = int(row[3])
            ymax = int(row[4])
            class_name = row[6]
            all += [[class_name, xmin, ymin, xmax, ymax]]

            add = [[img_name, [w, h, all]]]
            dumps += add


    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1
    
    # print('Statistics:')
    # pp(stat)
    # print('Dataset size: {}'.format(len(dumps)))
    
    return dumps
