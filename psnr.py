import numpy as np
from PIL import Image
import os

#Mean Square Error
def mse(A,B):
    tmp = 0
    m = width
    n = height
    for i in range (0,m):
        for j in range (0,n):
            tmp += (A[i,j]-B[i,j])**2   
    ms = float(tmp)/(m*n)
    return ms

# Calculate the peak signal-to-noise ratio
def psnr():
    ps = 20 * np.log10( 255 / np.sqrt(ms))
    return ps

for i in range (1,7):
    # file1 = input("This program is used to compare the differences between 2 images. \nPlease enter the name of original image :\n")
    file1 = str(i)+'.bmp'
    # file1 = '2.bmp'
    im1 = Image.open(file1)
    # file2 = input("Please enter the name of compressed image:\n")
    file2 = str(i)+'_8.gz.JPEG'
    # file2 = '2_8.gz.png'
    im2 = Image.open(file2)
    A = im1.load()
    B = im2.load()
    width,height = im1.size

    ms = mse(A,B)
    # print("MSE:",mse)
    p = psnr()
    print("PSNR of image",i," FOR 8:",p)
    # print("PSNR of image2:",psnr)
    
    file3 = str(i)+'_16.gz.JPEG'
    # file2 = '2_8.gz.png'
    im3 = Image.open(file3)
    C = im3.load()
    ms = mse(A,C)
    # print("MSE:",mse)
    p = psnr()
    print("PSNR of image",i," FOR 16:",p)
    # print("PSNR of image2:",psnr)

    origsize = os.path.getsize(file1)
    compsize = os.path.getsize(file2)
    compsize2 = os.path.getsize(file3)
    gz8 = os.path.getsize(file1[:-4]+"_8.gz")
    bz8 = os.path.getsize(file1[:-4]+"_8.bz2")
    xz8 = os.path.getsize(file1[:-4]+"_8.xz")
    gz16 = os.path.getsize(file1[:-4]+"_16.gz")
    bz16 = os.path.getsize(file1[:-4]+"_16.bz2")
    xz16 = os.path.getsize(file1[:-4]+"_16.xz")

    print(compsize,compsize2)

    data = [
    ["File Size in bytes","Compressed Size in bytes","gzip(8)","bzip2(8)","lzma(8)","gzip(16)","bzip2(16)","lzma(16)"],
    [origsize,compsize,gz8,bz8,xz8,gz16,bz16,xz16]
    ]
    print(data)