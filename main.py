import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder  # for detect the ball color
import pickle   # for labeling

cap             = cv2.VideoCapture('VelcroDartGameFiles/Videos/Video3.mp4')
frameCounter    = 0
cornerPoints    = [[377, 52], [944, 71], [261, 624], [1058, 612]]     # ini kordinat pixel ujung2 warna biru, soalnya gambarnya mau di pas in disini, kiriAtas=[377, 52], kananAtas=[944, 71], kiriBawah=[261, 624], kananBawah=[1058, 612]
colorFinder     = ColorFinder(False)
hsvVals         = {'hmin': 32, 'smin': 50, 'vmin': 0, 'hmax': 45, 'smax': 255, 'vmax': 255}
countHit        = 0
imgListBallsDetected = []
# hitDrawBallInfoList  = []
# totalScore      = 0

# with open('polygons', 'rb') as f:
#     polygonsWithScore = pickle.load(f)

# # print(polygonsWithScore)

def getBoard(img):
    width, height = int(400*1.5), int(380*1.5)  # unit milimeter
    pts1 = np.float32(cornerPoints)
    pts2 = np.float32([[0,0], [width,0], [0,height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)    # creating matrix
    imgOutput = cv2.warpPerspective(img, matrix, (width,height))    # transform the img
    # for x in range(4):
    #     cv2.circle(img, (cornerPoints[x][0], cornerPoints[x][1]), 15, (0,255,0), cv2.FILLED) # 15 is radius
    cv2.circle(img, (cornerPoints[0][0], cornerPoints[0][1]), 15, (255,0,0), cv2.FILLED)    # 15 is radius, BLUE
    cv2.circle(img, (cornerPoints[1][0], cornerPoints[1][1]), 15, (0,255,0), cv2.FILLED)    # 15 is radius, GREEN
    cv2.circle(img, (cornerPoints[2][0], cornerPoints[2][1]), 15, (0,0,255), cv2.FILLED)    # 15 is radius, RED
    cv2.circle(img, (cornerPoints[3][0], cornerPoints[3][1]), 15, (0,0,0), cv2.FILLED)      # 15 is radius, BLACK
    return imgOutput

def detetColorDarts(img):
    imgBlur = cv2.GaussianBlur(img, (7,7), 2)
    imgColor, mask = colorFinder.update(imgBlur, hsvVals)
    cv2.imshow('Image Mask 1', mask)

    # Decrease the noise
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Image Mask 2', mask)

    mask = cv2.medianBlur(mask, 9)  # 9 is high intensity
    cv2.imshow('Image Mask 3', mask)

    mask = cv2.dilate(mask, kernel, iterations=4)  # to join putih putih yg terpisah menjadi satu
    cv2.imshow('Image Mask 4', mask)

    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Image Mask 5', mask)

    cv2.imshow('Image Blur', imgBlur)
    cv2.imshow('Image Color', imgColor)
    

    return mask

while True:
    # success, img = cap.read()
    # cv2.imshow('Image', img)
    # cv2.waitKey(1)     # if you need to slow down, change from 1 to 50

    frameCounter += 1
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):   # CAP_PROP_FRAME_COUNT adalah jumlah frame dalam video tersebut (contoh 13 detik * 30 frame = 390 frame), kalau CAP_PROP_FRAME_COUNT itu dapetnya 404 frame berarti 13.467 detik
        frameCounter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    
    imgBoard = getBoard(img)
    mask = detetColorDarts(imgBoard)

    # ================ Detect the ball stuck to the velco dart board
    ### remove the previous Detection
    for x, img in enumerate(imgListBallsDetected):
        mask = mask - img
        cv2.imshow(str(x), mask)
    # ================ Detect the ball stuck to the velco dart board
    

    # find the contour of white are
    imgContours, conFound = cvzone.findContours(imgBoard, mask, 3500)   # 3500 is minimum area


    # ================ Detect the ball stuck to the velco dart board
    # the ball hits or not the board and wait for 5 iteration to make sure the ball hit and not fall down
    if conFound:
        countHit += 1
        if countHit == 5:
            imgListBallsDetected.append(mask)
            print("Hit Detected")
            countHit = 0
    # end ================ Detect the ball stuck to the velco dart board

    # cv2.imwrite('img.png', imgBoard)          # di pakai sekali
    # cv2.imwrite('imgBoard.png', imgBoard)     # di pakai sekali
    cv2.imshow('Image',         img)
    cv2.imshow('ImageBoard',    imgBoard)
    cv2.imshow('Image Mask',    mask)
    cv2.imshow('Image Contours',imgContours)
    cv2.waitKey(1)     # if you need to slow down, change from 1 to 50