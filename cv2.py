import numpy as np
import cv2


def camera_in():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Camera open failed!')
        return

    print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inversed = ~frame
        cv2.imshow('frame',frame)
        cv2.imshow('inversed',inversed)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()
# if __name__ == '__main__':
#     camera_in()
#######################################3


def video_in():
    cap = cv2.VideoCapture('stopwatch.avi')
    if not cap.isOpened():
        print('Video open failed!')
        return
    print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS',fps)
    delay = round(1000/fps)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        inversed = ~frame
        cv2.imshow('frame',frame)
        cv2.imshow('inversed',inversed)
        if cv2.waitKey(delay) == 27:
            break
    cv2.destroyAllWindows()
# if __name__ == '__main__':
#     video_in()             



def camera_in_video_out():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Camera open failed!')
        return
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'
    delay = round(1000/fps)
    outputVideo = cv2.VideoWriter('output.avi',fourcc,fps,(w,h))
    if not outputVideo.isOpened():
        print('File open failed!')
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inversed = ~frame
        outputVideo.write(inversed)
        cv2.imshow('frame',frame)
        cv2.imshow('inversed',inversed)
        if cv2.waitKey(delay) == 27:
            break
    cv2.destroyAllWindows() 
if __name__ == '__main__':
    camera_in_video_out()            
############################33
    
def drawLines():
    img = np.full((400,400,3),255,np.uint8)
    cv2.line(img,(50,50),(200,50),(0,0,255))
    cv2.line(img,(50,100),(200,100),(255,0,255),3)
    cv2.line(img,(50,150),(200,150),(255,0,0),10)
    cv2.line(img,(250,50),(350,100),(0,0,255),1,cv2.LINE_4)
    cv2.line(img,(250,70),(350,120),(255,0,255),1,cv2.LINE_8)
    cv2.line(img,(250,90),(350,140),(255,0,0),1,cv2.LINE_AA)
    cv2.arrowedLine(img,(50,200),(150,200),(0,0,255),1)
    cv2.arrowedLine(img,(50,250),(350,250),(255,0,255),1)
    cv2.arrowedLine(img,(50,300),(350,300),(255,0,0),1,cv2.LINE_8,0,0.05)
    cv2.drawMarker(img, (50,350), (0,0,255),  cv2.MARKER_CROSS)
    cv2.drawMarker(img, (100,350), (0,0,255), cv2.MARKER_TILTED_CROSS)
    cv2.drawMarker(img, (150,350), (0,0,255), cv2.MARKER_STAR)
    cv2.drawMarker(img, (200,350), (0,0,255),cv2.MARKER_DIAMOND)
    cv2.drawMarker(img, (250,350), (0,0,255), cv2.MARKER_SQUARE)
    cv2.drawMarker(img, (300,350), (0,0,255), cv2.MARKER_TRIANGLE_UP)
    cv2.drawMarker(img, (350,350),  (0,0,255), cv2.MARKER_TRIANGLE_DOWN)
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
#if __name__ == '__main__':
    #drawLines()


def drawLines():
    img = np.full((400,400,3),255,np.uint8)
    cv2.line(img,(50,50),(200,50),(0,0,255))
    cv2.line(img,(50,100),(200,100),(255,0,255),3)
    cv2.line(img,(50,150),(200,150),(255,0,0),10)
    cv2.line(img,(250,50),(350,100),(0,0,255),1,cv2.LINE_4)
    cv2.line(img,(250,70),(350,120),(255,0,255),1,cv2.LINE_8)
    cv2.line(img,(250,90),(350,140),(255,0,0),1,cv2.LINE_AA)
    cv2.arrowedLine(img,(50,200),(150,200),(0,0,255),1)
    cv2.arrowedLine(img,(50,250),(350,250),(255,0,255),1)
    cv2.arrowedLine(img,(50,300),(350,300),(255,0,0),1,cv2.LINE_8,0,0.05)
    cv2.drawMarker(img, (50,350), (0,0,255),  cv2.MARKER_CROSS)
    cv2.drawMarker(img, (100,350), (0,0,255), cv2.MARKER_TILTED_CROSS)
    cv2.drawMarker(img, (150,350), (0,0,255), cv2.MARKER_STAR)
    cv2.drawMarker(img, (200,350), (0,0,255),cv2.MARKER_DIAMOND)
    cv2.drawMarker(img, (250,350), (0,0,255), cv2.MARKER_SQUARE)
    cv2.drawMarker(img, (300,350), (0,0,255), cv2.MARKER_TRIANGLE_UP)
    cv2.drawMarker(img, (350,350),  (0,0,255), cv2.MARKER_TRIANGLE_DOWN)
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
#if __name__ == '__main__':
   #drawLines()

# def drawText2():
#     img = np.full((200, 640, 3), 255, np.uint8)

#     text = "Hello, OpenCV"
#     fontFace = cv2.FONT_HERSHEY_TRIPLEX
#     fontScale = 2.0
#     thickness = 1
#     sizeText, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
#     org = ((img.shape[1] - sizeText[0] // 2, (img.shape[0] + sizeText[1]) //2) 
#     cv2.putText(img, text, org, fontFace, fontScale, (255,0,0), thickness)
#     cv2.rectangle(img, org, (org[0] + sizeText[0], org[1] - sizeText[1]), (0, 255, 0), 1)
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     cv2.destroyAllwindows()
#     )
#if __name__ == '__main__':
#    drawText2()    