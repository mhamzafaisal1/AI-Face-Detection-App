import cv2
from random import randrange

# Load some pre-trained data on face frontals from open cv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choosing an image to detect faces in
img = cv2.imread('manyrdj.jpg')


# Now make the image black and white for the algorithm to understand
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# drawing the rectangles around the face to easily identify the faces
# looping through the data if there are more faces
for (x, y, w, h) in face_coordinates:
    # face_coordinates is an array itself so [0] is to select the first one or the first face
    # (x, y, w, h) = face_coordinates[0]
    # the last two parts are for the color (Blue-Green-Red, thickness of the rectangle)
    cv2.rectangle(img,  (x, y), (x+w, y+h),
                  (randrange(255), randrange(255), randrange(255)), 2) # rand range is a function being used to make sure different colors pop up every time for the face detection

# These are the coordinates of the face itself [top, bottom, left, right]
print(face_coordinates)


#
cv2.imshow("Hamza Face Detector", img)

# In python this command is used to keep the window open until a key is pressed to clear it otherwise the window quickly shows up and closes, it is hard to notice.
cv2.waitKey()

print('Hello world')
