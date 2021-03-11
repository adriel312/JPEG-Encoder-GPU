import cv2
import numpy as np
import math
from zigzag import *

threadsperblock = 8
blockspergrid = 128

#bloco para dividir a imagem em grupos de 8x8 pixels
block_size = 8

#tabelas obtidas na internet, estes são os coeficientes da tabela de quantização sugeridos pela norma padrão do JPEG.
#https://www.impulseadventure.com/photo/jpeg-quantization.html
#matriz de quantização luminancia
quant_matL = np.array([     [16,11,10,16,24,40,51,61],
                            [12,12,14,19,26,58,60,55],
                            [14,13,16,24,40,57,69,56],
                            [14,17,22,29,51,87,80,62],
                            [18,22,37,56,68,109,103,77],
                            [24,35,55,64,81,104,113,92],
                            [49,64,78,87,103,121,120,101],
                            [72,92,95,98,112,100,103,99]])

#matriz de quantização crominancia
quant_matC = np.array([     [17,18,24,47,99,99,99,99],
                            [18,21,26,66,99,99,99,99],
                            [24,26,56,99,99,99,99,99],
                            [47,66,99,99,99,99,99,99],
                            [99,99,99,99,99,99,99,99],
                            [99,99,99,99,99,99,99,99],
                            [99,99,99,99,99,99,99,99],
                            [99,99,99,99,99,99,99,99]])

#@cuda.jit
def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []
    bitstream = ''
    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1
    return bitstream

Y = cv2.imread('out_img/Y.bmp', cv2.IMREAD_GRAYSCALE)
Cb = cv2.imread('out_img/Cb.bmp', cv2.IMREAD_GRAYSCALE)
Cr = cv2.imread('out_img/Cr.bmp', cv2.IMREAD_GRAYSCALE)
#print (input_img.shape)
[h, w] = Y.shape
h = np.float(h)
w = np.float(w)

nbh = math.ceil(h/block_size)
nbh = np.int(nbh)

nbw = math.ceil(w/block_size)
nbw = np.int(nbw)

H = np.int(h)
W = np.int(w)

#matriz vazia com o tamanho da imagem
emp_Y = np.zeros((H,W))
emp_Cb = np.zeros((H,W))
emp_Cr = np.zeros((H,W))

emp_Y[0:H,0:W] = Y[0:H,0:W]
emp_Cb[0:H,0:W] = Cb[0:H,0:W]
emp_Cr[0:H,0:W] = Cr[0:H,0:W]
#print(emp_img)

for i in range(nbh):
    #calcula o indice da linha inicial e final
    row_ind_1 = i*block_size                
    row_ind_2 = row_ind_1+block_size
    for j in range (nbw):
        #calcula o indice da coluna inicial e final
        col_ind_1 = j*block_size                       
        col_ind_2 = col_ind_1+block_size

        blockY = emp_Y[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
        blockCb = emp_Cb[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
        blockCr = emp_Cr[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
        #print(block)
        #aplicando a DCT no bloco
        DCTY = cv2.dct(blockY)
        DCTCb = cv2.dct(blockCb)
        DCTCr = cv2.dct(blockCr)
        #divisão dos coeficientes da DCT pela quantização
        DCTY = (DCTY/quant_matL).astype(int)
        DCTCb = (DCTCb/quant_matC).astype(int)
        DCTCr = (DCTCr/quant_matC).astype(int)
        #reordena os coeficientes DCT em zigzag
        hmax = DCTY.shape[0]
        wmax = DCTY.shape[1]
        reord_output = np.zeros(hmax * wmax)

        zigzag[blockspergrid, threadsperblock](DCTY, hmax, wmax, reord_output)
        reorderedY = reord_output
        reord_output = np.zeros(hmax * wmax)
        zigzag[blockspergrid, threadsperblock](DCTCb, hmax, wmax, reord_output)
        reorderedCb = reord_output
        reord_output = np.zeros(hmax * wmax)
        zigzag[blockspergrid, threadsperblock](DCTCr, hmax, wmax, reord_output)
        reorderedCr = reord_output
        #remonta a matriz reordered para 8x8
        reshapedY = np.reshape(reorderedY, (block_size, block_size))
        reshapedCb = np.reshape(reorderedCb, (block_size, block_size))
        reshapedCr = np.reshape(reorderedCr, (block_size, block_size))
        #copia os indices do bloco atual para a matriz reshaped
        emp_Y[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedY
        emp_Cb[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedCb
        emp_Cr[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshapedCr

#cv2.imshow('encoded image', np.uint8(emp_Cb))
#cv2.imwrite('out_img/dct.bmp', np.uint8(emp_img))

arrangedY = emp_Y.flatten()
arrangedCb = emp_Cb.flatten()
arrangedCr = emp_Cr.flatten()

#gravando os dados codificados com RLE para um arquivo de texto
arrangedY = arrangedY.astype(int)
bitstreamY = get_run_length_encoding(arrangedY)

arrangedCb = arrangedCb.astype(int)
bitstreamCb = get_run_length_encoding(arrangedCb)

arrangedCr = arrangedCr.astype(int)
bitstreamCr = get_run_length_encoding(arrangedCr)

bitstream = str(emp_Y.shape[0]) + " " + str(emp_Y.shape[1]) + " " + bitstreamY + " " + bitstreamCb + " " + bitstreamCr + ";"
#gravando image.txt
file1 = open("image.txt","w")
file1.write(bitstream)
file1.close()

cv2.waitKey(0)
cv2.destroyAllWindows()