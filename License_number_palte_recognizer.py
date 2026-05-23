import cv2
import easyocr

numberPlate_cascade = "numberplate_haarcade.xml"
detector = cv2.CascadeClassifier(numberPlate_cascade)

img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = detector.detectMultiScale(
      img_gray,scaleFactor=1.05, minNeighbors=7,
      minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
print(plates)

for (x,y,w,h) in plates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plateROI = img_gray[y:y+h,x:x+w]
    cv2.imshow("Numberplate", plateROI)
    
reader = easyocr.Reader(['en'])
text = reader.readtext(plateROI)

if len(text) == 0:
    print(text)
    print(text[0][1])

cv2.putText(img, text[0][1], (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, 
(0, 255, 0), 2)

cv2.imshow('Output', img)
cv2.waitKey(0)