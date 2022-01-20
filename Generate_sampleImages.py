# Taking samples using haarcascade with opencv

import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3,640)
cam.set(4,500)

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input("Enter your name : ")

print("Taking samples, look at camera .......")
count = 0

while True:

    ret, img = cam.read()          # Read camera
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # changing images to gray
    faces = detector.detectMultiScale(gray_image, 1.3,5)

    # creating rectangle around the face with x,y coordinate
    for(x,y,w,h) in faces:
        count +=1
        # Save the image file to sample_folder
        cv2.imwrite("imagesAttendance/"+ str(face_id)+".jpg", gray_image[y:y+h, x:x+w])

    k = cv2.waitKey(100) & 0xff
    if k ==27 or count>=1:
        break

print("Sample taken now closing the program")
cam.release()
cv2.destroyAllWindows()







