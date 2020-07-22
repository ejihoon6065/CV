import numpy as np
import cv2
import random


def corner_harris():
    src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    harris = cv2.cornerHarris(src, 3, 3, 0.04)
    harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for y in range(harris_norm.shape[0]) :
        for x in range(harris_norm.shape[1]) :
            if harris_norm[y, x] > 120 :
                if (harris[y, x] > harris[y-1, x]) :

                    cv2.circle(dst, (x, y), 5, (0, 0, 255), 2)
                
    cv2.imshow('src', src)
    cv2.imshow('harris_norm', harris_norm)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

#corner_harris()



###################################################################################################

def find_homography():
    src1=cv2.imread('box.png',cv2.IMREAD_GRAYSCALE)
    src2=cv2.imread('box_in_scene.png',cv2.IMREAD_GRAYSCALE)
    
    if src1 is None or src2 is None:
        print('Image load failed')
        return
    orb=cv2.ORB_create()
    


    keypoints1,desc1=orb.detectAndCompute(src1,None)
    keypoints2,desc2=orb.detectAndCompute(src2,None)
    
    matcher=cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches=matcher.match(desc1,desc2)
    matches=sorted(matches,key=lambda x:x.distance)
    good_matches=matches[:50]





    dst=cv2.drawMatches(src1,keypoints1,src2,keypoints2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    pts1=np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2).astype(np.float32)
    pts2=np.array([keypoints2[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2).astype(np.float)
    
    H,_=cv2.findHomography(pts1,pts2,cv2.RANSAC)
    (h,w)=src1.shape[:2]
    corners1=np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2).astype(np.float32)
    corners2=cv2.perspectiveTransform(corners1,H)
    corners2=corners2+np.float32([w,0])
    
    cv2.imshow('src1',src1)
    cv2.imshow('src2',src2)
    cv2.polylines(dst,[np.int32(corners2)],True,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow('dst',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

find_homography()

