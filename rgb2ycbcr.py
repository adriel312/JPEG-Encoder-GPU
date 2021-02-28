import cv2
import numpy as np
from numba import cuda

@cuda.jit
def rgb2ycbcr(img, Y, Cb, Cr):
    Y[0][0] = 1
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Y[i,j] = 0 + (0.299)*(img[i,j][2]) + (0.587)*(img[i,j][1]) + (0.114)*(img[i,j][0])
            Cb[i,j] = 128 - (0.168736)*(img[i,j][2]) - (0.331264)*(img[i,j][1]) + (0.5)*(img[i,j][0])
            Cr[i,j] = 128 + (0.5)*(img[i,j][2]) - (0.418688)*(img[i,j][1]) - (0.081312)*(img[i,j][0])
    

img = cv2.imread('in_img/1024x768.bmp', 1)
img = (img.astype(float))
YCbCr = np.empty([img.shape[0],img.shape[1]], dtype = np.float32)

# move input data to the device
in_img = cuda.to_device(img)
YCbCr = cuda.to_device(YCbCr)

Y  = cuda.device_array_like(YCbCr)
Cb = cuda.device_array_like(YCbCr)
Cr = cuda.device_array_like(YCbCr)
# create output data on the device
out_img = np.empty((img.shape[0], img.shape[1], 3), float)

threadsperblock = 8
blockspergrid = 128
#blockspergrid = (img.size + (threadsperblock - 1)) // threadsperblock
rgb2ycbcr[blockspergrid, threadsperblock](in_img, Y, Cb, Cr)

# wait for all threads to complete
cuda.synchronize()

#junta as componentes da imagem
out_img[:,:,1] = Y.copy_to_host()
cv2.imwrite('out_img/Y.bmp', Y.copy_to_host())
out_img[:,:,0] = Cb.copy_to_host()
cv2.imwrite('out_img/Cb.bmp', Cb.copy_to_host())
out_img[:,:,2] = Cr.copy_to_host()
cv2.imwrite('out_img/Cr.bmp', Cr.copy_to_host())

#print(out_img)
#print (np.empty((img.shape[0], img.shape[1], 3), float))
#print (Y.copy_to_host())

cv2.imwrite('out_img/ycbcr.bmp', out_img)
#cv2.imshow('image',ycrbr)
#cv2.waitKey(0)
#cv2.destroyAllWindows()