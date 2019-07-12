import numpy as np
import math
import cv2
import os
#from scipy.special import expit
from utils.box import BoundBox, box_iou, prob_compare
from utils.box import prob_compare2, box_intersection
from utils.udacity_voc_csv import udacity_voc_csv
import json
	
_thresh = dict({
	'person': .2,
	'pottedplant': .1,
	'chair': .12,
	'tvmonitor': .13
})

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findBoxFromCsv(dumps,name):
	cvslist = []
	sz = len(dumps)
	for i in range(0,sz):
		img = dumps[i][0]
	# bug fix 
		if img == name:
			cvslist.append(dumps[i][1][2])
	return cvslist

# from accuracy.py, PO 377, not accepted yet
def calculateIOU(boxA,boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	 
	interArea = (xB - xA + 1) * (yB - yA + 1)
	
	dx = (xB-xA+1)
	dy = (yB - yA + 1)
	if (dx>=0) and (dy>=0):
		interArea=dx*dy
	else:
		interArea=0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea =(boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	totalArea=float(boxAArea + boxBArea - interArea)
 	#print interArea , totalArea
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / totalArea
 
	# return the intersection over union value
	return iou

def getIOUForOneBox(boxb,csvbox):
	ioum = 0.0
	midx = 0
	for i in range(0,len(csvbox)):
		boxa = csvbox[i]
		iou = calculateIOU(boxa,boxb)
		if iou > ioum:
			ioum = iou
			midx = i
	return ioum

def postprocess(self, net_out, im, save = True, dumps=[]):
	"""
	Takes net output, draw net_out, save to disk
	"""
	# meta
	meta = self.meta
	H, W, _ = meta['out_size']
	threshold = meta['thresh']
	C, B = meta['classes'], meta['num']
	anchors = meta['anchors']
	net_out = net_out.reshape([H, W, B, -1])
	resultsForJSON = []
	boxes = list()
	for row in range(H):
		for col in range(W):
			for b in range(B):
				bx = BoundBox(C)
				bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
				bx.c = expit(bx.c)
				bx.x = (col + expit(bx.x)) / W
				bx.y = (row + expit(bx.y)) / H
				bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W
				bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H
				classes = net_out[row, col, b, 5:]
				bx.probs = _softmax(classes) * bx.c
				bx.probs *= bx.probs > threshold
				boxes.append(bx)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)):
			boxes[i].class_num = c
		boxes = sorted(boxes, key = prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				if box_iou(boxi, boxj) >= .4:
					boxes[j].probs[c] = 0.


	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	# total: total box of ground truth
	# proposals: number of final output boxes
	# correct: number of correct boxex against ground truth
	re_proposals = 0
	re_total = 0
	re_correct = 0
	csvlist =[]
	csvbox = []	
	netbox = []	
	# read annotation
	impath = im.split('/')[-1]
	if self.FLAGS.loglevel > 0:
		csvlist = findBoxFromCsv(dumps,impath)
		for i in range(0,len(csvlist)):
			cbox = csvlist[i]
			carray = [cbox[0][1],cbox[0][2],cbox[0][3],cbox[0][4]]
			csvbox.append(carray)
		re_total = len(csvlist)


	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(C < 2)
		label += labels[max_indx] * int(C>1)
		if max_prob > threshold:
			re_proposals = re_proposals +1
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			cv2.rectangle(imgcv, 
				(left, top), (right, bot), 
				colors[max_indx], thick)
			mess = '{}'.format(label)
			# predicted box
			if self.FLAGS.loglevel > 0:
				barray = [left,top,right,bot]
				netbox.append(barray)
			cv2.putText(imgcv, mess, (left, top - 12), 
				0, 1e-3 * h, colors[max_indx],thick//3)
			if self.FLAGS.logjson == 1:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % max_prob), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})


		

	if not save: return imgcv
	outfolder = os.path.join(self.FLAGS.test, 'out') 
	img_name = os.path.join(outfolder, im.split('/')[-1])
	
	if self.FLAGS.logjson == 1:
		textJSON = json.dumps(resultsForJSON)
	textFile = os.path.splitext(img_name)[0] + ".json"
	gt_file = os.path.splitext(img_name)[0] + "_gt.jpg"

	ioumax = 0.0
	ioutotal = 0.0
	cths = 0.5
	ths_boxcenter = 50
	iouavg = 0
	P = 0.0
	Recall = 0.0
	IOU = 0.0

	# DarkNet YOLO2 training output
	# ref: https://timebutt.github.io/static/understanding-yolov2-training-output/
	# ref: validate_detector_recall() from Darknet/samples/detector.c 
	if self.FLAGS.loglevel > 0:
		for i in range(0,re_proposals): 
			boxb = netbox[i]
			iou = getIOUForOneBox(boxb,csvbox)
			ioutotal = ioutotal + iou
			if iou > cths:
				re_correct = re_correct+1
			if iou > ioumax:
				ioumax = iou
		if re_proposals>0:
			iouavg = ioutotal/re_proposals
			P = 100.*re_correct/re_proposals
		if re_total>0:
			Recall = 100.*re_correct/re_total
			IOU = iouavg*100/re_total
	if self.FLAGS.loglevel > 2:
		print('Img: {0}, Proposal:{1}, Correct: {2}, Total: {3}, IOU: {4}, Recall: {5}, Precision: {6}'.format(
			impath,
			re_proposals,
			re_correct,
			re_total,
			IOU,
			Recall,
			P))
	# save json file
	if self.FLAGS.logjson == 1:
		with open(textFile, 'w') as f:
			f.write(textJSON)
	
	cv2.imwrite(img_name, imgcv)
	# save groundtruth file along with the result file
	if self.FLAGS.loglevel > 0 and self.FLAGS.loggt > 0:
		imgcvgt = cv2.imread(im) 
		for i in range(0,len(csvbox)):
			b = csvbox[i]
			left  = b[0]
			right = b[2]
			top   = b[1]
			bot   = b[3]
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			cv2.rectangle(imgcvgt, 
				(left, top), (right, bot), 
				colors[0], thick)
			cv2.putText(imgcvgt, csvlist[i][0][0], (left, top - 12), 
				0, 1e-3 * h, colors[3],thick//3)

	if self.FLAGS.loglevel > 0 and self.FLAGS.loggt > 0:
		cv2.imwrite(gt_file, imgcvgt)
		
	return P, Recall
