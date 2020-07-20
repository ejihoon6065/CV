import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


src1 = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)  
src2 = cv2.imread('square2.bmp', cv2.IMREAD_GRAYSCALE)  

if src1 is None or src2 is None :
    print('Image load failed!')
    sys.exit()



dst1 = cv2.add(src1, src2)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)

dst5 = cv2.bitwise_and(src1, src2)
dst6 = cv2.bitwise_or(src1, src2)
dst7 = cv2.bitwise_xor(src1, src2)
dst8 = cv2.bitwise_not(src1)

plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title('src1')
plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title('src2')
plt.subplot(233), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('add')
plt.subplot(234), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('addWeighted')
plt.subplot(235), plt.axis('off'), plt.imshow(dst3, 'gray'), plt.title('subtract')
plt.subplot(236), plt.axis('off'), plt.imshow(dst4, 'gray'), plt.title('absdiff')

#plt.show()


plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title('src1')
plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title('src2')
plt.subplot(233), plt.axis('off'), plt.imshow(dst5, 'gray'), plt.title('bitwise_and')
plt.subplot(234), plt.axis('off'), plt.imshow(dst6, 'gray'), plt.title('bitwise_or')
plt.subplot(235), plt.axis('off'), plt.imshow(dst7, 'gray'), plt.title('bitwise_xor')
plt.subplot(236), plt.axis('off'), plt.imshow(dst8, 'gray'), plt.title('bitwise_not')

#plt.show()

# 두 사진의 크기가 다르면 numpy 에러가 발생


src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None :
    print('Image load failed!')
    sys.exit()


emboss = np.array([[-1, -1, 0],
                  [-1, 0, 1],
                  [0, 1, 1]], np.float32)

dst = cv2.filter2D(src, -1, emboss, delta =128)

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()