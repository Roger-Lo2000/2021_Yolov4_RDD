"""
├── yolo
│   ├── labels.txt
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
├── people.jpg
├── people_out.jpg
├── street.jpg
├── street_out.jpg
├── video.mp4
├── video_out.avi
├── yolo_image.py
└── yolo_video.py
 if program cant find yolo folder in main folder it will crash."""
 # example usage: python yolo_image.py -i street.jpg -o output.jpg
import argparse
import time
import glob
import cv2
import json
import numpy as np
import firebase_admin
from google.cloud import firestore



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input image file")
parser.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output image file. Write only the name, without extension.")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
        help="threshold for non maxima supression")
parser.add_argument("-T", "--time", type=str)
parser.add_argument("-f", "--filename", type=str)
args = vars(parser.parse_args())

CONFIDENCE_THRESHOLD = args["confidence"]
NMS_THRESHOLD = args["threshold"]
impath = args["input"]
times = args["time"]
filenames = args["filename"]
weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]
#print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))
lbls = list()
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect(imgpath, nn):
        conf = []
        hole_type = []
        image = cv2.imread(imgpath)
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
        nn.setInput(blob)
        start_time = time.time()
        layer_outs = nn.forward(layer)
        end_time = time.time()

        boxes = list()
        confidences = list()
        class_ids = list()

        for output in layer_outs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        url="https://storage.cloud.google.com/output-rdd/"+times+"/"+filenames+".jpeg?authuser=0"
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        db = firestore.Client()
        doc_ref = db.collection(times).document(filenames)
        doc = doc_ref.get()
        print("{}".format(doc.to_dict()))
        print((doc.to_dict())['location'])
        location = (doc.to_dict())['location']
        address = (doc.to_dict())['address']
       # print(location.address.split(", ")[-4])
        docl_ref = db.collection((doc.to_dict())['location']).document(filenames)
       # print(docl_ref)        
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                initx = boxes[i][0]
                inity = boxes[i][1]
                width = boxes[i][2]
                height = boxes[i][3]
                wi,he,ch = image.shape
                centerx = (initx + (width/2))/wi
                centery = (inity + (width/2))/he
                width = width/wi
                height = height/he
                if (width > 1):
                    width = 1
                if (height> 1):
                    height = 1 
                line = str(class_ids[i])+" "+str(round(centerx,3))+" "+str(round(centery,3))+" "+str(round(width,3))+" "+str(round(height,3))
                if (confidences[i] > 0.4):
                    image_org = cv2.imread(imgpath)
                    cv2.imwrite("./label/"+filenames+".jpg",image_org)
                    with open("./label/"+filenames+".txt",'a') as file:
                        file.write(line)
                        file.write('\n')
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
                cv2.putText(image, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                label = "Inference Time: {:.2f} ms".format(end_time - start_time)
                cv2.putText(image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                
                conf.append(str(round(confidences[i],3)))
                hole_type.append(lbls[class_ids[i]])
                conf_str = ", ".join(conf)
                hole_str = ", ".join(hole_type)

                db = firestore.Client()
                doc_ref = db.collection(times).document(filenames)
                doc_ref.update({
                    'hole': 'yes',
                    'type': hole_str,
                    'confidences': conf_str,
                    'url': url,
                    'address': address,
                    })
                docl_ref.set({
                    'hole': 'yes',
                    'type': hole_str,
                    'confidences': conf_str,
                    'url': url,
                    'address': address,
                    })

                
        else:
            db = firestore.Client()
            doc_ref = db.collection(times).document(filenames)
            doc_ref.update({ 
                'hole': 'no',
                'type': 'none',
                'confidences': '0',
                'url': url,
                'address': address,
               
                })
            docl_ref.set({
                'hole': 'no',
                'type': 'none',
                'confidences': 'o',
                'url': url,
                'address': address,
                })
         
        # cv2.imshow("image", image)
        if args["output"] != "":
            cv2.imwrite(args["output"], image)
        cv2.waitKey(0)

detect(impath, net)
time.sleep(1)
