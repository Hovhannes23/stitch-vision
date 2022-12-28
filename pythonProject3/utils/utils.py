import cv2
import numpy as np
import matplotlib.pyplot as plt


def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.THRESH_BINARY_INV, 1, 11, 2)
    ret, imgThresholdBinInvOtsu = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # showImage(imgThreshold)
    showImage(imgThresholdBinInvOtsu)
    return imgThresholdBinInvOtsu


def showImage(img):
    plt.imshow(img)
    plt.show()


def stackImages(imgs):
    for x in range(0, len(imgs)):
        if len(imgs[x]) == 2: imgs[x] = cv2.cvtColor(imgs[x], cv2.COLOR_GRAY2DGR)
    stack = np.hstack(imgs)
    return stack


def findBiggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) >= 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(points):
    points = points.reshape((points.shape[0], 2))
    pointsN = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    pointsN[0] = points[np.argmin(add)]
    pointsN[3] = points[np.argmax(add)]
    # pointsN[3] = np.array([2714, 3379])
    diff = np.diff(points, axis=1)
    pointsN[1] = points[np.argmin(diff)]
    pointsN[2] = points[np.argmax(diff)]
    # pointsN[2] = points[1]
    return pointsN


def change_num(num, divisor):
    num = int(num)
    divisor = int(divisor)
    whole_part = num // divisor
    num = int(whole_part * divisor)
    return num


def get_rectangle_points(points):
    X = points[:, :, 0]
    Y = points[:, :, 1]
    X_max = max(X)[0]
    X_min = min(X)[0]
    Y_max = max(Y)[0]
    Y_min = min(Y)[0]

    X_max = X_max - X_min
    X_min = 0
    Y_max = Y_max - Y_min
    Y_min = 0
    new_points = np.float32([
        [X_min, Y_min], [X_max, Y_min],
        [X_min, Y_max], [X_max, Y_max]
    ])

    return new_points


def splitBoxes(img):
    rows = np.vsplit(img, 7)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 53)
        for box in cols:
            boxes.append(box)
    return boxes


def cutSmallPiece(img):
    rows = np.vsplit(img, 10)
    pieces = []
    for r in rows:
        cols = np.hsplit(r, 1)
        for piece in cols:
            pieces.append(piece)
    return pieces


def split_into_cells(img, rows_num, columns_num):
    cells = []
    rows = np.vsplit(img, rows_num)
    for i, r in enumerate(rows, start=0):
        cells_in_row = np.hsplit(r, columns_num)
        for j, cell in enumerate(cells_in_row, start=0):
            cells.append(cell)
            # cv2.imwrite('result/{}_{}.png'.format(i, j), cell)
    cells = np.array(cells)
    # cells = np.reshape(cells, (rows_num, columns_num, cells.shape[1], cells.shape[2], cells.shape[3]))
    return cells
