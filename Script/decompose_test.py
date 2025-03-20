## 作用是获取某种分解方法的分解结果
from Decomposition.traditional import ssr, msr
import numpy as np
from PIL import Image




def decompose(method, I_path):

    if method == "ssr":
        return ssr(I_path)
    if method == 'msr':
        return msr(I_path)





if __name__ == '__main__':

    method = 'retinexd'
    test_num = 566

    if method != 'retinexd':

        input_path = f'Images/I/I_0/{test_num}.png'
        x, l, r = decompose(method, input_path)
        X = (x * 255).astype(np.uint8)
        L = (l * 255).astype(np.uint8)
        R = (r * 255).astype(np.uint8)

        X = Image.fromarray(X, mode='L')
        L = Image.fromarray(L, mode='L')
        R = Image.fromarray(R, mode='L')

        X = X.resize((256, 256))
        L = L.resize((256, 256))
        R = R.resize((256, 256))

        X.save(f'Results/msr_decompose/x_{test_num}.png')
        L.save(f'Results/msr_decompose/l_{test_num}.png')
        R.save(f'Results/msr_decompose/r_{test_num}.png')

    else:

        I_path = f'Images/I/I_0/{test_num}.png'
        S_path = f'Images/S/S_0/{test_num}.png'
        E_path = f'Images/E/E_0/{test_num}.png'

        I = Image.open(I_path).convert('L')
        S = Image.open(S_path).convert('L')
        E = Image.open(E_path).convert('L')

        I = I.resize((256, 256))
        S = S.resize((256, 256))
        E = E.resize((256, 256))

        I.save(f'Results/retinexd_decom/i_{test_num}.png')
        S.save(f'Results/retinexd_decom/s_{test_num}.png')
        E.save(f'Results/retinexd_decom/e_{test_num}.png')
        
        







        



