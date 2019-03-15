# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:33:17 2019

@author: John
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 
import pathlib
import pydicom
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon, resize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage import morphology
from skimage.segmentation import clear_border


p=pathlib.Path('c:/Users/Public/NIHCTs/')
rtheta= np.linspace(0., 180., 512, endpoint=False)
for x in p.iterdir():
    fx=os.path.abspath(x)
    if  x.match('000020.dcm'):
        ds=pydicom.dcmread(fx)
        image=ds.pixel_array
        plt.figure(dpi=300)
        plt.title(x.name)
        img1_plt=plt.imshow(image,cmap=plt.cm.bone)
        img1_plt.set_clim(1000.0,1080.0)
        plt.show()
        rad_image=radon(image,theta=rtheta,circle=True)
        plt.figure(dpi=300)
        plt.title('Sinogram of starting image')
        plt.imshow(rad_image,cmap=plt.cm.bone)
        plt.show()
        
rec1_image=iradon(rad_image,theta=rtheta,circle=True)
plt.figure(dpi=300)
img_plt=plt.imshow(rec1_image,cmap=plt.cm.bone)
img_plt.set_clim(1000.0, 1080.0)
plt.title('Simple reconstruction, brain window')
plt.show()

threshold=threshold_otsu(image)
bw=clear_border(morphology.closing(image>threshold,morphology.square(3)))

labelimage=label(bw,connectivity =1)

props=regionprops(labelimage)
area=[ele.area for ele in props ]
largest_blob_indx=np.argmax(area)
largest_blob_label=props[largest_blob_indx].label
no_interest_pixels=np.ones_like(image,dtype=np.uint8)
no_interest_pixels[labelimage == largest_blob_label] = 0
no_interest_pixel_image= no_interest_pixels * image
plt.figure(dpi=300)
plt.title('structures of no interest image')
plt.imshow(no_interest_pixel_image,cmap=plt.cm.bone)
plt.show()
no_interest_radon=radon(no_interest_pixel_image,theta=rtheta,circle=True)
plt.figure(dpi=300)
img_plt=plt.imshow(no_interest_radon,cmap=plt.cm.bone)
plt.title('Sinogram to subtract')
plt.show()
mod_radon=rad_image-no_interest_radon
plt.figure(dpi=300)
img_plt=plt.imshow(mod_radon,cmap=plt.cm.bone)
plt.title('Subtracted Sinogram')
plt.show()

rec2_image=iradon(mod_radon,theta=rtheta,circle=True)
plt.figure(dpi=300)
img_plt=plt.imshow(rec2_image,cmap=plt.cm.bone)
img_plt.set_clim(1000.0, 1080.0)
plt.title('Recon after sinogram subtraction')
plt.show()
difference_image=np.abs(rec1_image-rec2_image)
plt.figure(dpi=300)
img_plt=plt.imshow(difference_image,cmap=plt.cm.bone)
img_plt.set_clim(1000.0, 1080.0)
plt.title('Recon large difference image')
plt.show()
plt.figure(dpi=300)
img_plt=plt.imshow(difference_image,cmap=plt.cm.bone)
img_plt.set_clim(10.0, 100.0)
plt.title('Recon medium difference image')
plt.show()

plt.figure(dpi=300)
img_plt=plt.imshow(difference_image,cmap=plt.cm.bone)
img_plt.set_clim(1.0, 10.0)
plt.title('Recon small difference image')
plt.show()
