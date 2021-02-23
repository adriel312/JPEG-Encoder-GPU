import cv2
import numpy as np
from numba import cuda

@cuda.jit
def rgb2ycbcr(img, out_img):
    out_img[0][0] = 1
    #YCbCr_img = np.empty((img.shape[0], img.shape[1], 3), float)
    #Y = np.empty([img.shape[0],img.shape[1]], dtype = np.float32)
    #Cb = np.empty([img.shape[0],img.shape[1]], dtype = np.float32)
    #Cr = np.empty([img.shape[0],img.shape[1]], dtype = np.float32)
    

img = cv2.imread('in_img/1024x768.bmp', 1)
img = (img.astype(float))

# move input data to the device
in_img = cuda.to_device(img)
# create output data on the device
out_img = cuda.device_array_like(in_img)

threadsperblock = 8
blockspergrid = 128
#blockspergrid = (img.size + (threadsperblock - 1)) // threadsperblock
rgb2ycbcr[blockspergrid, threadsperblock](in_img, out_img)

# wait for all threads to complete
cuda.synchronize()

print(out_img.copy_to_host())

#cv2.imwrite('ycbcr.bmp', input_image)
#cv2.imshow('image',input_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()