import numpy as np
import cv2
import os
import argparse
from matplotlib import pyplot as plt

def ground_truth(gt_path):
    gt = cv2.imread(gt_path, 1)
    gt_grey = cv2.imread(gt_path, 0)
    dft_gt = cv2.dft(np.float32(gt_grey),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_gt = np.fft.fftshift(dft_gt)
    magnitude_spectrum_gt = 20*np.log(cv2.magnitude(dft_shift_gt[:,:,0],dft_shift_gt[:,:,1]))
    
    return gt, magnitude_spectrum_gt

def moire(moire_path):
    img = cv2.imread(moire_path, 1)
    img_grey = cv2.imread(moire_path, 0)
    dft_img = cv2.dft(np.float32(img_grey),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_img = np.fft.fftshift(dft_img)
    magnitude_spectrum_img = 20*np.log(cv2.magnitude(dft_shift_img[:,:,0],dft_shift_img[:,:,1]))

    return img, magnitude_spectrum_img


def color(img):
    
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    d = rows // 15
    
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-d:crow+d, ccol-d:ccol+d] = 1
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back


def fourier(img):
    blue, green, red = cv2.split(img)
    img_back_blue = color(blue)
    img_back_red = color(red)
    img_back_green = color(green)
    
    img_back_merge = cv2.merge((img_back_red, img_back_green, img_back_blue))
    img_back_gray = cv2.cvtColor(img_back_merge, cv2.COLOR_RGB2GRAY)
    
    dft_back = cv2.dft(np.float32(img_back_gray),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_back = np.fft.fftshift(dft_back)
    magnitude_spectrum_back = 20*np.log(cv2.magnitude(dft_shift_back[:,:,0],dft_shift_back[:,:,1]))

    return img_back_merge, magnitude_spectrum_back

def save(gt_path, gt, img, img_back_merge, magnitude_spectrum_gt, magnitude_spectrum_img, magnitude_spectrum_back):
    folder = gt_path.split('.')[0].split('\\')[-1]
    os.makedirs(f'test_result/{folder}', exist_ok = True)
    
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_back_merge = img_back_merge / img_back_merge.max()
    
    plt.figure(figsize=(10, 10))
    plt.subplot(321),plt.imshow(gt)
    plt.title('GT Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(322),plt.imshow(magnitude_spectrum_gt, cmap = 'gray')
    plt.title('GT spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(323),plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(324),plt.imshow(magnitude_spectrum_img, cmap = 'gray')
    plt.title('Input spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(325),plt.imshow(img_back_merge)
    plt.title('FT'), plt.xticks([]), plt.yticks([])
    plt.subplot(326),plt.imshow(magnitude_spectrum_back, cmap = 'gray')
    plt.title('FT spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.savefig(f'test_result/{folder}/spectrum.png')
    plt.imsave(f'test_result/{folder}/gt.jpg', gt)
    plt.imsave(f"test_result/{folder}/moire.jpg", img)
    plt.imsave(f"test_result/{folder}/ft.jpg", img_back_merge)
    
    
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_path", metavar='str', type=str)
    parser.add_argument("img_path", metavar='str', type=str)
    args = parser.parse_args()


    gt, magnitude_spectrum_gt = ground_truth(args.gt_path)
    img, magnitude_spectrum_img = moire(args.img_path)
    img_back_merge, magnitude_spectrum_back = fourier(img)
    save(args.gt_path, gt, img, img_back_merge, magnitude_spectrum_gt, magnitude_spectrum_img, magnitude_spectrum_back)