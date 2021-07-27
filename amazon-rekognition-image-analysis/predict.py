# 連番静止画に変換したいmp4ファイルと同じ場所にこのスクリプトを置く
# mp4ファイル名をvideoFileにセットする
# mp4の情報をframe_rate/width/heightにセットする
# python3 predict.py を実行

import boto3
import cv2
import glob
import os

# set Amazon Rekognition Custom Labels's project ARN
projectVersionArn = "<ARN>"

# set movie's information
videoFile = "cat.mp4"
frame_rate = 25
width = 1920
height = 1080

images = 'images'

def Video2Image():

    if not os.path.isdir(images):
        os.mkdir(images)

    # read movie file
    cap = cv2.VideoCapture(videoFile)
    
    # divide into the continuous still images and save them as PNG format image.
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        print("Processing frame id: {}".format(frameId))
        ret, frame = cap.read()
        if (ret != True):
            break
        name = images + '/image_' + str(int(frameId)).zfill(4) + '.png'
        cv2.imwrite(name, frame)
    
    cap.release()

def analyzeVideo():

    # settings for make video file
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter('result_' + videoFile, fourcc, frame_rate, (width, height))
    
    rekognition = boto3.client('rekognition')
    file_list = sorted(glob.glob(images + "/*"))

    # analyze each image
    for i, d in enumerate(file_list):
            
        frameId = i+1 #current frame number
        print("Processing frame id: {}".format(frameId))
        frame = cv2.imread(d)
        hasFrame, imageBytes = cv2.imencode(".png", frame)
        
        # analyze an image using Amazon Rekognition Custom Labels endpoint
        response = rekognition.detect_custom_labels(
            Image={
                'Bytes': imageBytes.tobytes(),
            },
            MinConfidence = 87,
            ProjectVersionArn = projectVersionArn
        )
    
        # draw rectangles on an image using the analysis results
        for output in response["CustomLabels"]:
            print(output)
            Name = output['Name']
            Confidence = output['Confidence']
            
            w = output['Geometry']['BoundingBox']['Width']
            h = output['Geometry']['BoundingBox']['Height']
            left = output['Geometry']['BoundingBox']['Left']
            top = output['Geometry']['BoundingBox']['Top']
            w = int(w * width)
            h = int(h * height)
            left = int(left*width)
            top = int(top*height)
            
            if Name == 'normal':
                cv2.rectangle(frame,(left,top),(left+w,top+h),(0,255,0),2)
                cv2.putText(frame, Name + ': ' + str(round(Confidence, 2)), (left+w+10, 200)
                , cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame,(left,top),(left+w,top+h),(0,0,255),2)
                cv2.putText(frame, Name + ': ' + str(round(Confidence, 2)), (left+w+10, 200)
                , cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255), 2, cv2.LINE_AA)
                           
        writer.write(frame)

    writer.release()

Video2Image()
analyzeVideo()