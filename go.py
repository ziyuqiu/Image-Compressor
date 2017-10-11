#package import
import numpy as np
from PIL import Image
import gzip
import bz2
import lzma

#####################################################################
#							  SET UP			  				    #
#####################################################################

# STEP 1 Adjust -----------------------------------------------------
# level adjustment, subtract 128 from each pixel	
	# srcPic: source picture
def adjust(srcPic,imwidth,imheight):
	pxls = srcPic.load()
	M = np.zeros([imheight,imwidth])
	for i in range (0,imheight):
		for j in range (0,imwidth):		
			M[i,j] = pxls[j,i]-128
	return M
# lreverse evel adjustment, plus 128 to each pixel
	#X: source picture
	#side: side
def re_adjust(X,side):
	A=np.zeros([side,side])
	for i in range (0,side):				
		for j in range (0,side):
			A[i,j]=X[i,j]+128
	return A

# STEP 2 2D-DCT -----------------------------------------------------
# generate 2D DCT matrix
	# side: side
def dctMatrix(side):
	Y=np.zeros([side,side])
	for p in range (0,side):		
		if (p==0):
			a=np.sqrt(1/side)
		else:
			a=np.sqrt(2/side)			
		for q in range (0,side):
			Y[p,q] = a*np.cos(np.pi*(2*q+1)*p/(2*side))	
	return Y #dct matrix
# apply 2D-DCT to matrix
	#X: source matrix
	#dct_M: DCT matrix
def dct(X,dct_M):
	return np.dot(np.dot(dct_M,X),np.transpose(dct_M))
# apply reverse 2D-DCT to matrix
	#X: source matrix
	#dct_M: DCT matrix
def reverse_DCT(X,dct_M):
	return np.dot(np.dot(np.transpose(dct_M),X),dct_M)

# STEP 3 Quantization -----------------------------------------------
# Decide Suggested lumininence quantization matrix
	#n: side
def lum(n):
	lum_for_8 = np.matrix('16 11 10 16 24 40 51 61;'+
					  '12 12 14 19 26 58 60 55;'+
					  '14 13 16 24 40 57 69 56;'+
					  '14 17 22 29 51 87 80 62;'+
					  '18 22 37 56 68 109 103 77;'+
					  '24 35 55 64 81 104 113 92;'+
					  '49 64 78 87 103 121 120 101;'+
					  '72 92 95 98 112 100 103 99')
	if (n==8):
		return lum_for_8 
	elif (n==16):		
		lum_for_16=np.zeros([16,16])
		for i in range (0,8):
			for j in range (0,8):
				lum_for_16[2*i,2*j]=lum_for_8[i,j]
				lum_for_16[2*i+1,2*j]=lum_for_8[i,j]
				lum_for_16[2*i,2*j+1]=lum_for_8[i,j]
				lum_for_16[2*i+1,2*j+1]=lum_for_8[i,j]
		return lum_for_16

# STEP 4 ENCODE -----------------------------------------------------
# DC coefficient dictionary
	# n: number to be encoded
	# return [category,length,code]
def dictionary(n):
	n = np.abs(n)
	if (n==0):
		return [0,2,'00']
	elif (n==1):
		return [1,3,'010']
	elif (2<=n<=3):
		return [2,3,'011']
	elif (4<=n<=7):
		return [3,3,'100']
	elif (8<=n<=15):
		return [4,3,'101']
	elif (16<=n<=31):
		return [5,3,'110']
	elif (32<=n<=63):
		return [6,4,'1110']
	elif (64<=n<=127):
		return [7,5,'11110']
	elif (128<=n<=255):
		return [8,6,'111110']
	elif (256<=n<=511):
		return [9,7,'1111110']
	elif (512<=n<=1023):
		return [10,8,'11111110']
	elif (1024<=n<=2047):
		return [11,9,'111111110']
	elif (2048<=n<=4095):
		return [12,10,'1111111110']
	elif (4096<=n<=8191):
		return [13,11,'11111111110']
	elif (8192<=n<=16383):
		return [14,12,'111111111110']
	elif (16384<=n<=32767):
		return [15,13,'1111111111110']

# decode dictionary
	# input: input string
	# i: index
	# return [decoded number,next index]
def decode_common(input,i):
	global flag
	length = 0
	j = 0
	if (input[i]=='0'):
		if (input [i+1]=='0'):
			cat = 0
			length = 2
		elif (input[i+1]=='1'):
			if (input [i+2]=='0'):
				cat =1
				length = 3
			if (input [i+2]=='1'):
				cat =2
				length = 3
	elif (input[i]=='1'):
		if (input[i+1]=='0'):
			if (input[i+2]=='0'):
				cat = 3
				length = 3
			elif (input[i+2]=='1'):
				cat = 4
				length = 3
		else:
			for j in range (1,12):
				if (i+j+1<=len(input) and input[i+j]=='1'):
					if(input[i+j+1]=='0'):
						cat = 4+j
						length = cat-2
						break
					elif(input[i+j+1]=='1' and j==10):
						flag = 1
						return
	bina = input[i+length:i+length+cat]
	num = bin_to_dec(bina)
	i = i+length+cat
	return [num, i]	

# STEP 5 zig-zag ----------------------------------------------------
# Zigzag Traverse
	# M: matrix to be traversed
	# n: side
	# return list of zigzag traversed matrix
def zigzag(M,n):
	index = -1
	Z=[0]*(M.size-1)
	for i in range (0, 2*(n-1)):
		if (i < n):
			bound = 0
		else:
			bound = i-n+1
		for j in range (bound,i-bound+1):
			index += 1
			if (i%2==1):
				Z[index] = M[j,i-j]
			else:
				Z[index] = M[i-j,j]
	return Z
# Reverse zig zag array to matrix
	#Z: zig zag array
	#n: side
def reverse_zz(Z,n):
	index = -1
	# initiate a new matrix
	M = np.zeros([n,n])
	for i in range (0, 2*(n-1)):
		if (i < n):
			bound = 0
		else:
			bound = i-n+1
		for j in range (bound,i-bound+1):
			index += 1
			if (i%2==1):
				M[j,i-j] = Z[index]
			else:
				M[i-j,j] = Z[index]
	return M

# OTHERS ------------------------------------------------------------
# Conversion from decimal to binary 
	#n: decimal number
	#w: width
	#return binary number of n
def binary(n,w):
	n=np.int(n)
	if (n>0):
		return np.binary_repr(n)
	elif (n<0):
		tmp = bin(n)[3:]
		tmp2 =''
		for i in range (0,len(tmp)):
			if (tmp[i]=='0'):
				tmp2+='1'
			elif (tmp[i]=='1'):
				tmp2+='0'
		return tmp2
	else:
		return ''
# Conversion from binary to deicimal
	#n: binary number
	#return decimal number of n
def bin_to_dec(n):
	if (n==''):
		return 0
	if (n[0]=='1'):
		return np.int(n,2)
	else:
		tmp =''
		for i in range (0,len(n)):
			if (n[i]=='0'):
				tmp+='1'
			elif (n[i]=='1'):
				tmp+='0'
		return 0-np.int(tmp,2)

# Functions for decoding --------------------------------------------
# dot multiplication
	#X: matrix 1
	#Y: matrix 2
	#side: side
def dot_mul(X,Y,side):
	D=np.zeros([side,side])
	for i in range (0,side):				
		for j in range (0,side):
			D[i,j]=X[i,j]*Y[i,j]
	return D

# return the difference between two matrixes for error check
def diff(A,B,width,height):
	Diff=np.zeros([width,height])
	for i in range (0,width):				
		for j in range (0,height):
			Diff[i,j]=A[i,j]-B[i,j]
	return Diff

# decode the DC coefficients
	#input: input string to be decoded
	#side: side
	#imwidth: image width
	#imheight: image height
def decode_dc(input,side,imwidth,imheight):
		index = 0
		i = 0
		dclst = [0]*int(imwidth*imheight/(side*side))
		while i <= (len(input)-1):
			result = decode_common(input,i)
			if(flag==0):
				num = result[0]
				i = result[1]
				if(index == 0):
					dclst[index] = num
				else:
					dclst[index] = dclst[index-1]+num
				index += 1
			else:
				return [dclst,i+20]

# decode the AC coefficients
	#input: input string to be decoded
	#side: side
	#zf: zfill number decided by side
	#DCresult: DC result list
	#imwidth: image width
	#imheight: image height
	#factor: factor decided by side for luminance quantization matrix
def decode_ac(input,side,zf,DCresult,imwidth,imheight,factor):
	index = 0
	i = 0
	lstcount = 0
	lst = [0]*int(imwidth*imheight/(side*side))
	inlst = [0]*(side*side-1)
	while i <= (len(input)-1):				
		zeros = np.int(input[i:i+zf],2)
		for k in range (0,zeros):
			if (index<=side*side-2):
				inlst[index] = 0
			index += 1
		if(index > (side*side-2)):
			lst[lstcount] = tran(inlst,lstcount,side,DCresult,factor)
			lstcount +=1
			inlst = [0]*(side*side-1)
			index = 0
		i += zf
		tmp = decode_common(input,i)
		num = tmp[0]
		i = tmp[1]
		if(index <= (side*side-2)):
			inlst[index] = num
			index +=1
		if(index >(side*side-2)):
			lst[lstcount] = tran(inlst,lstcount,side,DCresult,factor)
			lstcount +=1
			inlst = [0]*(side*side-1)
			index = 0
	return lst

#transition during decode process
	#lst: matrix in array
	#n: index of sub image
	#side: side
	#DCresult: DC result list
	#factor: image quality factor
def tran(lst,n,side,DCresult,factor):
	dct_M = dctMatrix(side)
	Q = lum(side)
	Q = Q/factor
	BqNew = reverse_zz(lst,side)
	BqNew[0,0] = DCresult[0][n]
	Bnew =  dot_mul(BqNew,Q,side)
	ASnew = np.rint(reverse_DCT(Bnew,dct_M))
	Anew = re_adjust(ASnew,side)
	return Anew

# put the sub images back to a whole picture
	#imwidth: image width
	#imheight: image height
	#side: side
	#ACresult: AC result list
def recon(imwidth,imheight,side,ACresult):
	k = 0
	outim = np.zeros([imheight,imwidth])
	for i in range (0,imheight,side):
		for j in range (0,imwidth,side):
			outim[i:i+side,j:j+side] = ACresult[k]
			k += 1
	return outim

#####################################################################
#							    ENCODE				  			    #
#####################################################################

def encode():
	#Step 0 import image
	fileName = input("Please enter the name of file that you want to compress:\n")
	image = Image.open(fileName)
	imwidth, imheight = image.size
	imwidthcode = bin(imwidth)[2:].zfill(12)
	imheightcode = bin(imheight)[2:].zfill(12)
	print("Image size:", imwidth, "*", imheight)
	gray = image.convert("L") #convert to gray-scale images

	quality = input("Please enter the quality of PSNR you want. Please enter L for LOW, M for MEDIUM, H for HIGH:\n")
	if(quality == 'L'): #30
		factor = 1
		qualitycode = '00'
	elif(quality == 'M'): #40
		factor = 10
		qualitycode = '01'
	elif(quality == 'H'): #50
		factor = 90
		qualitycode = '10'
	side = int(input("Do you want to slice the image into 8*8 or 16*16 tiles? Please enter 8 or 16\n"))
	print('Compressing...Please wait')

	# decide luminance quantization and zfill number based on size
	if side == 8:
		zf = 6
		sidecode = 0
	elif side == 16:
		zf = 8
		sidecode = 1
	Q = lum(side)/factor
	dct_M = dctMatrix(side)
	# level adjustment, subtract 128 from each pixel
	pixels=adjust(gray,imwidth,imheight)
	# Partition the image into 8*8 OR 16*16 sub-images and do transformation
	k = 1
	#DC initiation
	DC_lst = [0]*int(imwidth*imheight/(side*side))
	origLst = [0]*side*side*int(imwidth*imheight/(side*side))
	DC_output = ''
	dc_base = pixels[0,0]
	dc_prev = 0
	#AC initiation
	AC_lst = [0]*int(imwidth*imheight/(side*side))
	AC_output = ''
	#crop
	for i in range (0,imheight,side):
		for j in range (0,imwidth,side):
			AS = pixels[i:i+side,j:j+side]
			origLst[k-1] = AS
			# Perform independent 2D DCT's on each of these sub-images
			B = dct(AS,dct_M)
			# Quantize by dividing by lum_quantization_matrix
			Bq = np.rint(B/Q)
			# Display entries of the quantized matrix in zig-zag order
			Z = zigzag(Bq,side)			
			# represent the DC coefficients by differences			
			dc_curr = Bq[0,0]
			diff = dc_curr-dc_prev
			result = dictionary(diff)
			DC_output += result[2] + binary(diff,result[1])
			DC_lst[k-1] = dc_curr
			dc_prev = dc_curr
			# AC coefficients
			AC_lst[k-1] = Z
			count = 0			
			for a in range (0, len(Z)):
				if(Z[a]!=0 or a==len(Z)-1):
					cnt = bin(count)[2:].zfill(zf)
					dic = dictionary(Z[a])
					AC_output += cnt + dic [2] + binary (Z[a], dic [1])
					count = 0
				else:
					count += 1							
			k += 1
	# output compressed data			
	
	encoded = str(sidecode) + str(qualitycode) + str(imwidthcode) + str(imheightcode) + DC_output	
	for w in range (0,20):
		encoded += '1'
	encoded += AC_output

	f = open("compressedData.txt","w")
	f.write(encoded)
	f.close

	# compress with lossless functions
	code = bytearray(encoded,'utf8')
	# gzip
	fileName = fileName[:-4]
	with gzip.open(fileName + '_' + str(side) + '.gz', 'w') as fg:
		fg.write(code)
	fg.close()
	# bzip2	
	tarbz2contents = bz2.compress(code,9)
	fb = open(fileName + '_' + str(side) + '.bz2', "wb")
	fb.write(tarbz2contents)
	fb.close()
	# lzma
	with lzma.open(fileName + '_' + str(side) + '.xz', "w") as fl:
		fl.write(code)
	fl.close()

	print("Your file is compressed into 3 ways.")
	# plt.imshow(pixels, cmap='gray', interpolation='nearest')
	# plt.show()

#####################################################################
#							    DECODE				  			    #
#####################################################################

def decode():
	fileName = input("Please enter the name of the file that you want to decompress:\n")
	if(fileName.endswith('.gz')):
		with gzip.open(fileName,'rt') as fg:
			filecontent = fg.read()
	elif(fileName.endswith('.bz2')):
		filecontent = bz2.open(fileName,'rt').read()
	elif(fileName.endswith('.xz')):
		filecontent = lzma.open(fileName,'rt').read()
	print('Decompressing...Please wait')

	code = filecontent
	if(code[0] == '0'):
		side = 8
		zf = 6
	elif(code[0] == '1'):
		side = 16
		zf = 8
	if(code[1:3]=='00'):
		factor = 1
	elif(code[1:3]=='01'):
		factor = 10
	if(code[1:3]=='10'):
		factor = 90

	imwidth = np.int(code[3:15],2)
	imheight = np.int(code[15:27],2)
	print('Image size:',imwidth,'*',imheight)
	DCresult = decode_dc(code[27:],side,imwidth,imheight)
	ACresult = decode_ac(code[27+DCresult[1]:len(code)],side,zf,DCresult,imwidth,imheight,factor)
	outMtx = recon(imwidth,imheight,side,ACresult)

	im = Image.fromarray(np.uint8(outMtx))
	im.show()
	im.save(fileName + '.JPEG')

#####################################################################
#							  PURPOSE			  				    #
#####################################################################
#controller
def ask(n):
	if (n == 0):
		print('Welcome to the magic image compressor!')
	else :
		done = input("Are you all set? Please enter Y or N:")
		if (done == 'Y' or done == 'y'):
			print('Goodbye!')
			return
	choice = input('What do you want to do?\n1 Compress an image\n2 Decompress an image\nPlease enter 1 or 2:\n')
	if (choice == '1'):
		encode()
		ask (n+1)
	else:		
		decode()
	n += 1

flag = 0
ask(0)

