import cv2
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    print(type(frame[0,0,0]))
    ff=cv2.resize(frame,(1000,1000))
    cv2.imshow('frame',ff)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break