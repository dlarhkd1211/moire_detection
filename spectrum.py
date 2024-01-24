import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def spectrum(sp):
    os.makedirs(f'img_test_spectrum/{sp}/gt', exist_ok=True)
    os.makedirs(f'img_test_spectrum/{sp}/moire', exist_ok=True)
    path = glob(f'img_test_original/{sp}/*/*')

    for pth in tqdm(path):
        img_grey = cv2.imread(pth, 0)
        dft_img = cv2.dft(np.float32(img_grey),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift_img = np.fft.fftshift(dft_img)
        magnitude_spectrum_img = 20*np.log(cv2.magnitude(dft_shift_img[:,:,0],dft_shift_img[:,:,1]))
        plt.imsave(pth.replace('original','spectrum'), magnitude_spectrum_img)
    print(f'{sp} convert to spectrum done!')
    
if __name__ == '__main__':
    spectrum('train')
    # spectrum('valid')