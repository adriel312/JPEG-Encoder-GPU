from numba import cuda

@cuda.jit
def zigzag (input, hmax, wmax, output):
    #w = width h = height
    h = 0
    w = 0

    hmin = 0
    wmin = 0

    i = 0

    while((h < hmax) and (w < wmax)):
        if((w + h) % 2) == 0:   #subindo
            
            if(h == hmin):                          #se estivermos na primeira linha
                output[i] = input[h, w]
                if(w == wmax):
                    h = h + 1
                else:
                    w = w + 1

                i = i + 1

            elif ((w == wmax-1) and (h < hmax)):    #se estivermos na ultima coluna
                output[i] = input[h, w]
                h = h + 1
                i = i + 1
        
            elif ((h > hmin) and (w < wmax-1)):     #qualquer um dos outros casos
                output[i] = input[h, w]
                h = h - 1
                w = w + 1
                i = i + 1

        else:   #descendo
            if ((h == hmax-1) and (w <= wmax-1)):   #se estivermos na ultima linha
                output[i] = input[h, w]
                w = w + 1
                i = i + 1
            
            elif (w == wmin):                       #se estivermos na primeira coluna
                output[i] = input[h, w]
                if (h == hmax-1):
                    w = w + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h < hmax-1) and (w > wmin)):     #qualquer um dos outros casos
                output[i] = input[h, w]
                h = h + 1
                w = w - 1
                i = i + 1

        if ((hmax == hmax-1) and (w == wmax-1)):    #ultimo pixel, inferior direito
            output[i] = input[h, w]
            break