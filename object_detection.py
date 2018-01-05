#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: igor
"""

# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

import argparse

#The object detection function
def detect(frame, net, transform):
    height,width = frame.shape[:2]
    
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))

    #Get the y value from the convolutional network    
    y = net(x)
    detections = y.data
    #detections = [batch, number of classes, number of occurence, (score,x0,y0,x1,y1)]

    scale = torch.Tensor([width,height,width,height])

    for i in range(detections.size(1)):
	#Get the label tag from the labelmap 
        label = labelmap[i - 1]

	#Try to detect all or the labels that are on the arguments list
        if ("all" in args.objects) or (label in args.objects):
            j = 0
            #Use a threshold of 60% for a correct qualification
            while detections[0, i, j, 0] >= 0.6:
                #Get the points of the rectangle of the object and converts it to numpy
                point = (detections[0,i,j,1:] * scale).numpy()
                #Draw a rectangle around the detected object
                cv2.rectangle(frame,(int(point[0]),int(point[1])),
                              (int(point[2]),int(point[3])),
                              (255,0,0), 2)
                #Create a lable of the rectangle
                cv2.putText(frame,label, (int(point[0]),int(point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2,cv2.LINE_AA)
                j += 1
            
    return frame

#Defining arguments for the program

parser = argparse.ArgumentParser(description='Object Detector')

parser.add_argument('--video', dest='parameter', nargs='?',
                   const='video', default='cam',
                   help='Flag to use the named video, it defaults to webcam')

parser.add_argument('objects', default = 'all', nargs='?', 
                   help='A list of objects label that it will be identified in the video, it defaults to all')

args = parser.parse_args()
print(args.parameter)

args.objects = args.objects.split(",")
print(args.objects)        

#Creating the SSD network    
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',
                               map_location = lambda storage, loc: storage))

#Creating the transformation for the network
transform = BaseTransform(net.size,(104/256.0, 117/256.0, 123/256.0))

#Doing the Object Detection on a video
if(args.parameter != 'cam'):
    reader = imageio.get_reader(args.parameter)
    #Get the frames per second from the reader metadata
    fps = reader.get_meta_data()['fps']

    #The output will be written in the 'output.mp4' file
    writer = imageio.get_writer('output.mp4',fps = fps)
    for i, frame in enumerate(reader):
        frame = detect(frame,net.eval(),transform)
        writer.append_data(frame)
        print("frame: " + str(i))   
    writer.close()
    
#Doing some Object Detection on the webcam
elif(args.parameter == 'cam'):
    reader = imageio.get_reader('<video0>')
    for i, frame in enumerate(reader,-1):
        frame = detect(frame,net.eval(),transform)
        cv2.imshow('Video',frame)

	#Press 'q' to close the webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            reader.close()
            break

    cv2.destroyAllWindows()

#Option to use opencv library to handle video input from webcam
    
#    videoCapture = cv2.VideoCapture(0)
#    while True:
#        _,frame = videoCapture.read()
#        frame = detect(frame,net.eval(),transform)
#        cv2.imshow('Video',frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break    
#    videoCapture.release()
#    cv2.destroyAllWindows()  
