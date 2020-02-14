import cv2
import numpy as np

def nothing(x):
    pass

    #track bar(to reach hsv)
cv2.namedWindow("Track_bar")
     #lower
cv2.createTrackbar("L_h", "Track_bar", 0, 360, nothing)
cv2.createTrackbar("L_s", "Track_bar", 0, 255, nothing)
cv2.createTrackbar("L_v", "Track_bar", 0, 255, nothing)
    #upper
cv2.createTrackbar("U_h", "Track_bar",   0, 360, nothing)
cv2.createTrackbar("U_s", "Track_bar", 255, 360, nothing)
cv2.createTrackbar("U_v", "Track_bar", 255, 360, nothing)

cap = cv2.VideoCapture(0)


while True :
    _, im = cap.read()
    l_h = cv2.getTrackbarPos("L_h","Track_bar")
    l_s = cv2.getTrackbarPos("L_s", "Track_bar")
    l_v = cv2.getTrackbarPos("L_v", "Track_bar")
    u_h = cv2.getTrackbarPos("U_h", "Track_bar")
    u_s = cv2.getTrackbarPos("U_s", "Track_bar")
    u_v = cv2.getTrackbarPos("U_v", "Track_bar")
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(im,lower,upper)


    cv2.imshow('mask' ,mask)
    key=cv2.waitKey(1)
    if key==27:
        break


cap.release()
cv2.destroyAllWindows()



