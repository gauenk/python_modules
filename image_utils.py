import numpy as np
import cv2

def overflowOnly(coordinate,rows,cols):
    if 0 > coordinate[0]: coordinate[0] = np.abs(coordinate[0])
    elif rows < coordinate[0]: coordinate[0] = rows - coordinate[0]
    if 0 > coordinate[1]: coordinate[1] = np.abs(coordinate[1])
    elif cols < coordinate[1]: coordinate[1] = cols - coordinate[1]

def zeroInTheRegion(coordinate,rows,cols):
    if 0 <= coordinate[0] and coordinate[0] <= rows: coordinate[0] = 0
    if 0 <= coordinate[1] and coordinate[1] <= cols: coordinate[1] = 0

def correctTranslatedIndex(coordinate,rows,cols):
    zeroInTheRegion(coordinate,rows,cols)
    overflowOnly(coordinate,rows,cols)

def getRotationScale(M,rows,cols):
    a = np.array([cols,0,1])
    b = np.array([0,0,1])
    ta = np.matmul(M,a)
    tb = np.matmul(M,b)
    correctTranslatedIndex(ta,rows,cols)
    correctTranslatedIndex(tb,rows,cols)
    scale_a_0 = rows / ( 2. * np.abs(ta[0]) + rows )
    scale_a_1 = rows / ( 2. * np.abs(ta[1]) + rows )
    scale_b_0 = cols / ( 2. * np.abs(tb[0]) + cols )
    scale_b_1 = cols / ( 2. * np.abs(tb[1]) + cols )
    scale_list = [scale_a_0,scale_a_1,scale_b_0,scale_b_1]
    scale = np.min([scale_a_0,scale_a_1,scale_b_0,scale_b_1])
    return scale

def getRotationInfo(angle,cols,rows):
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1.0)
    scale = getRotationScale(rotationMat,rows,cols)
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    return rotationMat,scale

def rotateImage(img,angle):
    # print('angle',angle)
    if angle is False:
        return img,None
    im_shape = img.shape
    rows,cols = img.shape[:2]
    rotationMat, scale = getRotationInfo(angle,cols,rows)
    img = cv2.warpAffine(img,rotationMat,(cols,rows),scale)
    rotateInfo = [angle,cols,rows,im_shape]
    return img,rotateInfo

def rotateImageList(imageList,angle):
    rot_image_list = []
    if type(imageList) is list:
        for image_index,image in enumerate(imageList):
            rot_image,_ = rotateImage(image,angle)
            rot_image_list.append(rot_image)
    else:
        is_single_bw_image_bool = (len(imageList.shape) == 2)
        is_single_color_image_bool = (len(imageList.shape) == 3) and (imageList.shape[2] == 3)
        if is_single_bw_image_bool or is_single_color_image_bool:
            print("actually single image; not list")
            rot_image,_ = rotateImage(imageList,angle)
            return rot_image
        for image_index in range(imageList.shape[0]):
            image = np.squeeze(imageList[image_index,...])
            rot_image,_ = rotateImage(image,angle)
            rot_image_list.append(rot_image)
    return rot_image_list

def saveImageList(imageList,prefix_name="save_image",label_string=None):
    if type(imageList) is list:
        for image_index,image in enumerate(imageList):
            if label_string is not None:
                filename = "{}_{}_{}.png".format(prefix_name,image_index,label_string[image_index])
            else:
                filename = "{}_{}.png".format(prefix_name,image_index)
            cv2.imwrite(filename,image)

    else:
        for image_index in range(imageList.shape[0]):
            image = np.squeeze(imageList[image_index])
            if label_string is not None:
                filename = "{}_{}_{}.png".format(prefix_name,image_index,label_string[image_index])
            else:
                filename = "{}_{}.png".format(prefix_name,image_index)
            cv2.imwrite(filename,image)

            
            
