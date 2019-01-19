# -*- coding: utf-8 -*-
import dlib
import time
import cv2

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


image = cv2.imread("7.JPG")
img_height, img_width = image.shape[:2]
BoxA = [210, 279, 210+285, 279+285]
BoxA[0] = int(BoxA[0]/img_width*300)
BoxA[1] = int(BoxA[1]/img_height*300)
BoxA[2] = int(BoxA[2]/img_width*300)
BoxA[3] = int(BoxA[3]/img_height*300)
image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml') 
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
hog_face_detector = dlib.get_frontal_face_detector()
net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb","opencv_face_detector.pbtxt")

print("Execution Time (in seconds) :")

#----------OPENCV---------------------------
start = time.time()
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False) 
net.setInput(blob)
detections = net.forward()
bboxes = []
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        x1 = int(detections[0, 0, i, 3] * 300)
        y1 = int(detections[0, 0, i, 4] * 300)
        x2 = int(detections[0, 0, i, 5] * 300)
        y2 = int(detections[0, 0, i, 6] * 300)
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
        break;
end = time.time()
print("OPENCV : ", format(end - start, '.2f'))
BoxB = [x1,y1,x2,y2]
IOU = bb_intersection_over_union(BoxA,BoxB)
print("IOU:",IOU)

#----------HOG----------------
start = time.time()
faces_hog = hog_face_detector(image, 1)
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
end = time.time()
print("HOG : ", format(end - start, '.2f'))
BoxB = [x,y,x+w,y+h]
IOU = bb_intersection_over_union(BoxA,BoxB)
print("IOU:",IOU)

#--------HAAR------------------
start = time.time()
faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5);
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
end = time.time()
print("Haar : ", format(end - start, '.2f'))
BoxB = [x,y,x+w,y+h]
IOU = bb_intersection_over_union(BoxA,BoxB)
print("IOU:",IOU)

#--------LBP------------------
start = time.time()
faces = lbp_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5);
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (120, 120, 0), 2)
end = time.time()
print("LBP : ", format(end - start, '.2f'))
BoxB = [x,y,x+w,y+h]
IOU = bb_intersection_over_union(BoxA,BoxB)
print("IOU:",IOU)


#--------CNN------------------
start = time.time()
faces_cnn = cnn_face_detector(image, 1)

for i, d in enumerate(faces_cnn):
    face = d.rect
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,120,120), 2)
    print(i)

end = time.time()
print("CNN : ", format(end - start, '.2f'))
BoxB = [x,y,x+w,y+h]
IOU = bb_intersection_over_union(BoxA,BoxB)
print("IOU:",IOU)



cv2.putText(image, "OPENCV", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,0,0), 2)
cv2.putText(image, "HOG", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)
cv2.putText(image, "HAAR", (img_width-50,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
cv2.putText(image, "LBP", (img_width-50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120,120,0), 2)
cv2.putText(image, "CNN", (img_width-50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,120,120), 2)

# display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()
cv2.destroyAllWindows()