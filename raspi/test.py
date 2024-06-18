import cv2 as cv
# open camera
cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L)

# set dimensions
cap.set(cv.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1440)

# take frame
ret, frame = cap.read()
# write frame to file
cv.imwrite('image.jpg', frame)
# release camera
cap.release()
