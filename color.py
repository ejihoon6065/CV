import numpy as np
import cv2
import math



def color_op():
    src = cv2.imread('butterfly.jpg', cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        print('Image load failed!')
        return

    print('src.shape:', src.shape)
    print('src.dtype:', src.dtype)

    print('The pixel value [B, G, R] at (0, 0) is', src[0,0])

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

#color_op()

######################################################################################

def color_inverse():
    src = cv2.imread('butterfly.jpg', cv2.IMREAD_COLOR)
    
    if src is None:
        print('Image load failed!')
        return

    dst = np.zeros(src.shape, src.dtype)

    for j in range(src.shape[0]) :
        for i in range(src.shape[1]) :
            p1 = src[j, i]
            p2 = dst[j, i]

            p2[0] = 255 - p1[0]
            p2[1] = 255 - p1[1]
            p2[2] = 255 - p1[2]
    
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

#color_inverse()

######################################################################################

def color_split():
    src = cv2.imread('candies.png', cv2.IMREAD_COLOR)
    
    if src is None:
        print('Image load failed!')
        return

    bgr_planes = cv2.split(src)

    cv2.imshow('src', src)
    cv2.imshow('B_plane', bgr_planes[0])
    cv2.imshow('G_plane', bgr_planes[1])
    cv2.imshow('R_plane', bgr_planes[2])
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    color_split()


######################################################################################

