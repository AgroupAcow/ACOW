import dlib
import time
import cv2

image = cv2.imread("0.JPG")
img_height, img_width = image.shape[:2]

hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb","opencv_face_detector.pbtxt")

print("Execution Time (in seconds) :")
start = time.time()
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
 
net.setInput(blob)
detections = net.forward()
bboxes = []
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        x1 = int(detections[0, 0, i, 3] * img_width)
        y1 = int(detections[0, 0, i, 4] * img_height)
        x2 = int(detections[0, 0, i, 5] * img_width)
        y2 = int(detections[0, 0, i, 6] * img_height)
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
end = time.time()
print("OPENCV : ", format(end - start, '.2f'))

start = time.time()

# apply face detection (hog)
faces_hog = hog_face_detector(image, 1)

end = time.time()
print("HOG : ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

start = time.time()

# apply face detection (cnn)
faces_cnn = cnn_face_detector(image, 1)

end = time.time()
print("CNN : ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

     # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.putText(image, "OPENCV", (img_width-50,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,0,0), 2)

cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
# display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()
cv2.destroyAllWindows()