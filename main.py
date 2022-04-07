import cv2
import mediapipe as mp
import numpy as np


def lerp(a, b, c):
    return int((c * a) + ((1 - c) * b))


def largestBox(boxes):
    """
    Given a list of boxes, return the box with the largest width

    :param boxes: a list of 4-tuples (x, y, w, h)
    :return: The largest box
    """

    lrg_width = 0
    lrg_box = None
    for box in boxes:
        if box[2] > lrg_width:
            lrg_box = BoundingBox(box[0], box[1], box[2], box[3])
            lrg_width = box[2]
    


    if lrg_box is None:
        # return original box
        lrg_box = BoundingBox(0, 0, 0, 0)

    return lrg_box


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.dim = [x, y, w, h]

    def lerpShape(self, newBox):
        for i in range(2):
            self.dim[i] = lerp(self.dim[i], newBox.dim[i], 0.4)
        for i in range(2):
            j = i + 2
            self.dim[j] = lerp(self.dim[j], newBox.dim[j], 0.7)


class Frame:

    boxIsVisible = False
    def __init__(self, img, box):
        self.zoom = 0.4
        self.img = img
        self.box = box
        x, y, w, h = box.dim
        self.postFilterBox = BoundingBox(x, y, w, h)

    def setZoom(self, amount):
        self.zoom = min(max(amount, 0.01), 0.99)

    def filter(self):

        # Declare basic variables
        screenHeight = self.img.shape[0]
        screenWidth = self.img.shape[1]
        screenRatio = float(screenWidth) / screenHeight

        (boxX, boxY, boxW, boxH) = self.box.dim
        distX1 = boxX
        # dist refers to the distances in front of and
        distY1 = boxY
        distX2 = screenWidth - distX1 - boxW        # behind the face detection box
        # EX: |---distX1----[ :) ]--distX2--|
        distY2 = screenHeight - distY1 - boxH

        # Equalize x's and y's to shortest length
        if distX1 > distX2:
            distX1 = distX2
        if distY1 > distY2:
            distY1 = distY2

        distX = distX1      # Set to an equal distance value
        distY = distY1

        # Trim sides to match original aspect ratio
        centerX = distX + (boxW / 2.0)
        centerY = distY + (boxH / 2.0)
        distsRatio = centerX / centerY

        if screenRatio < distsRatio:
            offset = centerX - (centerY * screenRatio)
            distX -= offset
        elif screenRatio > distsRatio:
            offset = centerY - (centerX / screenRatio)
            distY -= offset

        # Make screen to box ratio constant
        # (constant can be changed as ZOOM in main.py)
        if screenWidth > screenHeight:
            distX = min(0.5 * ((boxW / self.zoom) - boxW), distX)
            distY = min(
                ((1.0 / screenRatio) * (distX + (boxW / 2.0))) - (boxH / 2.0), distY)
        else:
            distY = min(0.5 * ((boxH / self.zoom) - boxH), distY)
            distX = min((screenRatio * (distY + (boxH / 2.0))) -
                        (boxW / 2.0), distX)

        # Crop image to match distance values
        newX = int(boxX - distX)
        # This is a debugging tool that prints the values of the new bounding box.
        newY = int(boxY - distY)
        newW = int(2 * distX + boxW)
        newH = int(2 * distY + boxH)

        # print(newX, newY, newW, newH)
        # if the new box is out of bounds, don't crop
        if not (newX < 0 or newY < 0 or newW > screenWidth or newH > screenHeight):
            self.crop([newX, newY, newW, newH])

        # Resize image to fit original resolution
        resizePercentage = float(screenWidth) / newW
        self.img = cv2.resize(self.img, (screenWidth, screenHeight))
        for i in range(4):
            self.postFilterBox.dim[i] = int(self.postFilterBox.dim[i] * resizePercentage)


        # Flip Filtered image on y-axis
        self.img = cv2.flip(self.img, 2)

    def drawBox(self):
        (x, y, w, h) = self.postFilterBox.dim
        if x > 0:
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    def crop(self, dim):
        x, y, w, h = dim
        self.img = self.img[y:y + h, x:x + w]
        self.postFilterBox.dim[0] -= x
        self.postFilterBox.dim[1] -= y

    def show(self):
        if self.boxIsVisible:
            self.drawBox()
        
        # reduce the size of the image by a factor of 2 for upsamling
        upres = cv2.resize(self.img, (0, 0), fx=0.3, fy=0.3)
        upres = sr.upsample(upres) # upsample image
        upres = cv2.resize(upres, (1280, 720))

        cv2.imshow("Before", self.img)
        cv2.imshow("Face-Tracking", upres)


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

ZOOM = 0.22             # Medium = 0.2 to 0.3,    Close = 0.35 to 0.5
SHOW_BOX = False         # Show face detection box around the largest detected face
SCALE_FACTOR = 1.17     # Medium = 1.2,     Close = 1.14
MIN_NEIGHBORS = 8       # 8
MINSIZE = (60, 60)    # Medium = (60, 60),    Close = (120, 120)

cap = cv2.VideoCapture(0)

# Create global detection box for steady screen transformation
box = BoundingBox(-1, -1, -1, -1)

'''
Experimental Function - Super Resolution using FSRCNN
'''

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "./models/FSRCNN_model.pb"

sr.readModel(path)
sr.setModel("fsrcnn", 3)



with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.66) as face_detection:
    prev_results = []
    counter = 0
    while cap.isOpened():
        _, image = cap.read()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        FACES = []

        if results.detections:
            for detection in results.detections:
                x1, y1, w, h = detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height
                x1 = int(x1 * image.shape[1])
                y1 = int(y1 * image.shape[0])
                w = int(w * image.shape[1])
                h = int(h * image.shape[0])

                FACES.append([x1, y1, w, h])

        RESULTS = np.array(FACES)

        
        if RESULTS.size > 0:
            
            if counter % 5 == 0:
                prev_results.append(RESULTS)
            
            boxLarge = largestBox(RESULTS)
            if box.dim[0] == -1:
                box = boxLarge
            else:
                box.lerpShape(boxLarge)

            counter += 1
        
        else:
            # If no faces are detected, zoom out from prev_result to [-1, -1, -1, -1]
            if len(prev_results) > 0:
                box.lerpShape(largestBox(prev_results[-1]))
                counter += 1
            else:   
                box.dim = [-1, -1, -1, -1]
                counter = 0




        frame = Frame(image, box)
        frame.boxIsVisible = SHOW_BOX
        frame.setZoom(ZOOM)

        frame.filter()
        box = frame.box

        frame.show()

        # Stop if escape key is pressed
        k = cv2.waitKey(30)
        if k == 27:
            break
        if k == 49:
            SHOW_BOX = not SHOW_BOX
        if k == 50:
            ZOOM = max(ZOOM - 0.05, 0.01)
        if k == 51:
            ZOOM = min(ZOOM + 0.05, 0.99)


cap.release()
