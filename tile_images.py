import numpy as np
from PIL import Image,ImageFont,ImageDraw

def add_image_boarder(image):
    image[0,:,:] = 255
    image[-1,:,:] = 255
    image[:,0,:] = 255
    image[:,-1,:] = 255
    
def __get_nrows_and_ncols(image_list,nrows,ncols):
    nimgs = len(image_list)
    if nrows is None and ncols is None:
        nrows = int(np.sqrt(nimgs))
        ncols = int(nimgs / nrows) + 1 * (nimgs % nrows != 0)
    if nrows is None: nrows = int(nimgs / ncols) + 1 * (nimgs % ncols != 0)
    if ncols is None: ncols = int(nimgs / nrows) + 1 * (nimgs % nrows != 0)
    assert nrows * ncols >= nimgs, "rows and cols are not set correctly"
    return nrows,ncols

def __get_tile_shape(yshapes,xshapes,nrows,ncols,ysep,xsep,color):
    height = sum(sorted(yshapes,reverse=True)[:nrows]) + nrows * ysep
    width = sum(sorted(xshapes,reverse=True)[:ncols]) + ncols * xsep
    shape = [height,width,color]
    return shape

def __get_boundary_from_index(index,shapes,sep):
    if index == 0: start = 0
    else: start = index * (shapes[index-1]+sep)
    end = start + shapes[index]
    return start,end

def __set_color(image_list,color_bool):
    if color_bool: return 3
    if len(image_list[0].shape) < 3: return 1
    return image_list[0].shape[2]


def __mange_image(image,show_boarder):
    if len(image.shape) < 3:
        image = image[:,:,np.newaxis]
    if show_boarder: 
        add_image_boarder(image)
    return image

def __mange_tile(tile,show_boarder):
    if show_boarder: 
        add_image_boarder(tile)


def __get_location(ysep,ystart,yend,xsep,xstart,xend,location):
    if location in ['left','top']: raise ValueError("Invalid location")
    if location == 'right':
        yTxtStart = (yend - ystart) * .1  + ystart
        xTxtStart = xend + (xend-xstart)*.2
        txtStart = (xTxtStart,yTxtStart)
    if location == 'bottom':
        yTxtStart = yend
        xTxtStart = (xend - xstart) * .5  + xstart
        txtStart = (xTxtStart,yTxtStart)
    return txtStart

def __add_label(draw,label,ysep,ystart,yend,xsep,xstart,xend,location):
    txtStart = __get_location(ysep,ystart,yend,xsep,xstart,xend,location)
    font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/TlwgTypewriter-Bold.ttf", 16)
    #draw.text(txtStart,label,fill=(255,255,255,255),font=font)
    draw.text(txtStart,label,fill=255,font=font)

def __convert_numpy_to_image(image):
    if len(image.shape) < 3 or image.shape[2] == 1:
        img = Image.fromarray(np.squeeze(image),'L')    
    else:
        img = Image.fromarray(image,'RGB')    
    return img

def __normalize_image_list(image_list):
    jimage_list = []
    for image in image_list:
        if image.max() == 0: continue
        nimage = (image/image.max() * 255.).astype(np.uint8)
        jimage_list.append(nimage)
    return jimage_list

def create_tile(image_list,xsep=10,ysep=10,nrows=None,ncols=None,color_bool=None,label_list=None,show_boarder=False,label_location='bottom'):
    """
    xsep: # number of pixels between images along x dim
    ysep: # number of pixels between images along y dim
    """
    image_list = __normalize_image_list(image_list)

    nrows,ncols = __get_nrows_and_ncols(image_list,nrows,ncols)
    yshapes = [image.shape[0] for image in image_list]
    xshapes = [image.shape[1] for image in image_list]
    color = __set_color(image_list,color_bool)
    shape = __get_tile_shape(yshapes,xshapes,nrows,ncols,ysep,xsep,color)
    tile = np.zeros(shape,dtype=np.uint8)
    
    nimages = len(image_list)
    image_index = 0
    for yindex in range(nrows):
        y_start,y_end = __get_boundary_from_index(yindex,yshapes,ysep)
        if nimages <= image_index: break
        for xindex in range(ncols):
            x_start,x_end = __get_boundary_from_index(xindex,xshapes,xsep)
            image = image_list[image_index]
            tile[y_start:y_end,x_start:x_end,:] = __mange_image(image,show_boarder)
            image_index += 1
            if nimages <= image_index: break
    __mange_tile(tile,show_boarder)
    
    # add labels
    image_index = 0
    if label_list is not None and len(label_list) > 0:
        Image_image = __convert_numpy_to_image(tile)
        draw = ImageDraw.Draw(Image_image)
        for yindex in range(nrows):
            ystart,yend = __get_boundary_from_index(yindex,yshapes,ysep)
            if nimages <= image_index: break
            for xindex in range(ncols):
                xstart,xend = __get_boundary_from_index(xindex,xshapes,xsep)
                label = label_list[image_index]
                __add_label(draw,label,ysep,ystart,yend,xsep,xstart,xend,label_location)
                image_index += 1
                if nimages <= image_index: break
        tile = np.array(Image_image)
    return tile
