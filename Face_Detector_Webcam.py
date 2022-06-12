import cv2
from random import randrange

# Load some pre-trained data on face frontals from open cv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choosing the video/webcam to detect faces in, now using video capture it will look for videos and if 0 is passed to it them it defaults to a webcam.
webcam = cv2.VideoCapture(0)


# iterate forever over frames until the video ends
while True:

    # read the current frame
    successful_frame_read, frame = webcam.read()
    # Now make the image black and white for the algorithm to understand
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # drawing the rectangles around the face to easily identify the faces
    # looping through the data if there are more faces
    for (x, y, w, h) in face_coordinates:
        # face_coordinates is an array itself so [0] is to select the first one or the first face
        # (x, y, w, h) = face_coordinates[0]
        # the last two parts are for the color (Blue-Green-Red, thickness of the rectangle)
        cv2.rectangle(frame,  (x, y), (x+w, y+h),
                      (0, 255, 0), 2)  # rand range is a function being used to make sure different colors pop up every time for the face detection

    # These are the coordinates of the face itself [top, bottom, left, right]
    cv2.imshow("Hamza Face Detector", frame)

    # In python this command is used to keep the window open until a key is pressed to clear it otherwise the window quickly shows up and closes, it is hard to notice.
    # using the variable key to store what key is being pressed and then using it later on
    key = cv2.waitKey(1)

    # to quit the app by using the key "Q", the key is being fetched from the waitKey function and the ASCII characters are being compared
    if key == 81 or key == 113:
        break


# to release the videocapture object
webcam.release()
