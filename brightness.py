import numpy as np
import cv2

def brightness1() :
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    dst = cv2.add(src, 100)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows

# brightness1()


def brightness2() :
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    dst = np.empty(src.shape, src.dtype) # shape과 dtype으로 만듦.
    for y in range(src.shape[0]):       # for문을 y부터 돌림
        for x in range(src.shape[1]):
            dst[y,x] = src[y,x] + 100


    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows

# brightness2()    


def brightness4() :
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    def update(pos):
        dst = cv2.add(src, pos)
        cv2.imshow('dst', dst)

    cv2.namedWindow('dst')
    cv2.createTrackbar('Brightness', 'dst', 0, 100, update)
    update(0)

    cv2.waitKey()
    cv2.destroyAllWindows


# if __name__ == '__main__':
#     brightness4()

def saturated(value) :
    if value > 255:
        value = 255
    elif value < 0 :
        value = 0

    return value

    def brightness3() :
        src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    dst = np.empty(src.shape, dtype = src.dtype)
    for y in range(src.shape[0]):       # for문을 y부터 돌림
        for x in range(src.shape[1]):
            dst[y,x] = saturated(src[y,x] + 100)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows

def contrast1():
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None :
        print('Image load failed!')
        return

    s = 2.0
    dst = cv2.multiply(src, s)
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows


def contrast2():
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None :
        print('Image load failed!')
        return

    alpha = 1.0
    dst = np.clip(src + (src - 128.)*alpha, 0, 255).astype(np.uint8)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows

# if __name__ == '__main__':
#      contrast1()
#      contrast2()
##########################################################################################################################

def calcGrayHist(img) :
    channels = [0]
    histSize = [256]
    histRange = [0, 256]

    hist = cv2.calcHist([img], channels, None, histSize, histRange)
    return hist

def getGreyHistImage(hist) :
    histMax = np.max(hist)

    imgHist = np.full((100, 256), 255, dtype=np.uint8)          # why 255??
    for x in range(256) :
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0]*100 / histMax))
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist

def histogram_stretching():
    src = cv2.imread('hawkes.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None :
        print('Image load failed!')
        return

    gmin = float(np.min(src))
    gmax = float(np.max(src))

    dst = ((src - gmin)*255. / (gmax - gmin)).astype(np.uint8)

    cv2.imshow('src', src)
    cv2.imshow('srcHist', getGreyHistImage(calcGrayHist(src)))

    cv2.imshow('dst', dst)
    cv2.imshow('dstHist', getGreyHistImage(calcGrayHist(dst)))

    cv2.waitKey()
    cv2.destroyAllWindows


def histogram_equalization():
    src = cv2.imread('hawkes.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None :
        print('Image load failed!')
        return

    dst = cv2.equalizeHist(src)
    cv2.imshow('src', src)
    cv2.imshow('srcHist', getGreyHistImage(calcGrayHist(src)))

    cv2.imshow('dst', dst)
    cv2.imshow('dstHist', getGreyHistImage(calcGrayHist(dst)))

    cv2.waitKey()
    cv2.destroyAllWindows

if __name__ == '__main__':
    histogram_stretching()      
    histogram_equalization()