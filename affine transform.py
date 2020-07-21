import numpy as np
import cv2


def affine_transform():
    src =cv2.imread('tekapo.bmp')
    if src is None:
        print('Image load failed!')
        return

    rows = src.shape[0]
    cols = src.shape[1]
    src_pts = np.array([[0,0],
                        [cols -1, 0],
                        [cols -1, rows -1]]).astype(np.float32)
    dst_pts = np.array([[50,50],
                        [cols -100, 100],
                        [cols -50, rows -50]]).astype(np.float32)   

    affine_mat = cv2.getAffineTransform(src_pts, dst_pts)
    dst = cv2.warpAffine(src, affine_mat, (0,0))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows

# if __name__ == '__main__':
#     affine_transform()    

###############################일반 변환 ################################

def affine_translation() :
    src =cv2.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    affine_mat = np.array([[1,0, 150],
                        [0, 1, 100]]).astype(np.float32)

    dst = cv2.warpAffine(src, affine_mat, (0,0))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows

# if __name__ == '__main__':
#     affine_translation()  


###############################전단 변환 ################################
def affine_shear() :
    src =cv2.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    rows = src.shape[0]
    cols = src.shape[1]

    mx = 0.3
    affine_mat = np.array([[1, mx, 0],
                        [0, 1, 0]]).astype(np.float32)

    dst = cv2.warpAffine(src, affine_mat, (int(cols + rows * mx), rows))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows    

# if __name__ == '__main__':
#     affine_shear()


############################### 전단 변환 ################################
def affine_scake() :
    src =cv2.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    dst1 = cv2.resize(src, (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    dst2 = cv2.resize(src, (1920, 1280))
    dst3 = cv2.resize(src, (1920,1280), interpolation=cv2.INTER_CUBIC)
    dst4 = cv2.resize(src, (1920,1280), interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow('src', src)
    cv2.imshow('dst1', dst1[400:800, 500:900])
    cv2.imshow('dst2', dst2[400:800, 500:900])
    cv2.imshow('dst3', dst3[400:800, 500:900])
    cv2.imshow('dst4', dst4[400:800, 500:900])
    cv2.waitKey()
    cv2.destroyAllWindows      
    
# if __name__ == '__main__':
#     affine_scake()



############################### 회전 변환 ################################

def affine_rotation() :
    src =cv2.imread('tekapo.bmp')

    if src is None:
        print('Image load failed!')
        return

    cp = (src.shape[1] / 2, src.shape[0]/ 2)
    affine_mat = cv2.getRotationMatrix2D(cp, 20, 1)

    dst = cv2.warpAffine(src, affine_mat, (0,0))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    
    cv2.waitKey()
    cv2.destroyAllWindows 

# if __name__ == '__main__':
#     affine_rotation()


############################### 대칭 변환 ################################

def affine_flip() :
    src = cv2.imread('eastsea.bmp')
    if src is None:
        print('Image load failed!')
        return

    cv2.imshow('src', src)
    
    for flipCode in [1, 0, -1]:
        dst = cv2.flip(src, flipCode)

        desc = 'flipCode : %d' % flipCode

        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('dst', dst)
        cv2.waitKey()

    cv2.destroyAllWindows 

# if __name__ == '__main__':
#     affine_flip()

############################### 대칭 변환 ################################

import sys

def on_mouse(event, x, y, flags, param) :
    global cnt, src_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x,y]).astype(np.float32)
            cnt += 1
            cv2.circle(src, (x,y), 5, (0, 0, 255), -1)
            cv2.imshow('src', src)

        if cnt == 4:
            w = 200
            h = 300

            dst_pts = np.array([[0, 0],
                               [w -1, 0],
                               [w-1, h-1],
                               [0, h-1]]).astype(np.float32)

            pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            dst = cv2.warpPerspective(src, pers_mat, (w,h))  
            cv2.imshow('dst', dst)

cnt = 0
src_pts = np.zeros([4, 2], dtype = np.float32)
src = cv2.imread('card.bmp')

if src is None :
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('src')
cv2.setMouseCallback('src', on_mouse)


cv2.imshow('src', src)  
cv2.waitKey(0)
cv2.destroyAllWindows  

    
if __name__ == '__main__':
    on_mouse(5, 1, 1, flags, param)
