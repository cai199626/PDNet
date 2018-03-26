import cv2
import numpy as np
#import cPickle as pickle
import h5py
import random
#PIC_PATH='/home/zcb/datasets/trainSet/Stimuli/Alldata/'
#SALMAP_PATH='/home/zcb/datasets/trainSet/FIXATIONMAPS/Alldata/'

#PIC_PATH='/home/cx/keras_saliency/datasets/ecssd/images/0/'
#SALMAP_PATH='/home/cx/keras_saliency/datasets/ecssd/ground_truth_mask/0/'
#SALCON  10k eye fix
#PIC_PATH = '/home/zcb/datasets/SALCON/images/train_image/train_images/'
#SALMAP_PATH = '/home/zcb/datasets/SALCON/train/train_masks/'
PIC_PATH = '/home/cx/keras_saliency/datasets/DUTS-TR/DUTS-TR-Image/'
SALMAP_PATH = '/home/cx/keras_saliency/datasets/DUTS-TR/DUTS-TR-Mask/'

VAL_PATH = '/home/cx/keras_saliency/datasets/DUTS-TE/DUTS-TE-Image/'
VALMASK_PATH = '/home/cx/keras_saliency/datasets/DUTS-TE/DUTS-TE-Mask/'



PIC_PATH2 = '/home/cx/keras_saliency/datasets/MSRA10K_Imgs_GT/Imgs/'
SALMAP_PATH2 = '/home/cx/keras_saliency/datasets/MSRA10K_Imgs_GT/Masks/'
#SALMAP_PATH = '/home/zcb/datasets/MSRA10K_Imgs_GT/saliencymap/'



datalist = open(PIC_PATH+'list.txt','r')
namelist=[l.strip('\n') for l in datalist.readlines()]

sallist = open(SALMAP_PATH+'list.txt','r')
sallist=[l.strip('\n') for l in sallist.readlines()]

datalist2 = open(PIC_PATH2+'list.txt','r')
namelist2=[l.strip('\n') for l in datalist2.readlines()]

sallist2 = open(SALMAP_PATH2+'list.txt','r')
sallist2=[l.strip('\n') for l in sallist2.readlines()]

val_datalist = open(VAL_PATH+'list.txt','r')
val_namelist=[l.strip('\n') for l in val_datalist.readlines()]

val_sallist = open(VALMASK_PATH+'list.txt','r')
val_sallist=[l.strip('\n') for l in val_sallist.readlines()]



input_h=224
input_w=224
output_h=224
output_w=224
NumSample=len(namelist)
NumSample2=len(namelist2)
val_num = len(val_namelist)

X1 = np.zeros((NumSample+NumSample2,input_h, input_w,3), dtype='float32')
Y1 = np.zeros((NumSample+NumSample2,output_h,output_w,1), dtype='uint8')
#Y1 = np.zeros((NumSample,output_h,output_w,1), dtype='float32')
NumAll = NumSample+NumSample2
print NumAll

print val_num
val_num = 2500
VAL_X = np.zeros((val_num,input_h, input_w,3), dtype='float32')
VAL_Y = np.zeros((val_num,output_h,output_w,1), dtype='uint8')
 
for i in range(val_num):
    img = cv2.imread(VAL_PATH+val_namelist[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print img.shape
    img = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
    img=img.astype(np.float32)/255.
    VAL_X[i]=img
    
    label = cv2.imread(VALMASK_PATH+val_sallist[i],cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(output_w,output_h),interpolation=cv2.INTER_CUBIC)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
    VAL_Y[i]=label.reshape(output_h,output_w,1)



for i in range(NumSample):
#name1 = namelist[i][0:namelist[i].index('.')]
#   name2 = sallist[i][0:sallist[i].index('_')]
#   if name1 != name2: 
#       print error
    img = cv2.imread(PIC_PATH+namelist[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print img.shape
    img = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
    #print img.shape
    #cv2.imshow('show',img)
    img=img.astype(np.float32)/255.
    #img = img /255. 
    #img -= 1.
    if(cmp(img.shape , (input_h,input_w,3)) == 0):
        #img = img.transpose(2,0,1).reshape(3, input_h, input_w)
        X1[i]=img
    else:
        print 'error'
    
    #np.set_printoptions(threshold='nan') 
    label = cv2.imread(SALMAP_PATH+sallist[i],cv2.IMREAD_GRAYSCALE)
    #label = loadSaliencyMapSUN(names[i])
    label = cv2.resize(label,(output_w,output_h),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('label',label)
    #cv2.waitKey(0)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
#	print 'data',X1[i]
#	print 'label',label
    #Y1.append(label.reshape(1,48*48))
    Y1[i]=label.reshape(output_h,output_w,1)

for i in range(NumSample2):
    img = cv2.imread(PIC_PATH2+namelist2[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print img.shape
    img = cv2.resize(img,(input_w,input_h),interpolation=cv2.INTER_CUBIC)
    img=img.astype(np.float32)/255.
    X1[i+NumSample]=img
    
    label = cv2.imread(SALMAP_PATH2+sallist2[i],cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label,(output_w,output_h),interpolation=cv2.INTER_CUBIC)
    label = label.astype(np.float32)
    label /=255
    label[label > 0.5]=1
    label[label <=0.5]=0
    label=label.astype(np.uint8)
    Y1[i+NumSample]=label.reshape(output_h,output_w,1)

#random.seed(1)
#rand=range(NumAll)
#random.shuffle(rand)
#
#split_at = int(NumAll * 0.88)
#x , y = X1[rand[0:split_at]],Y1[rand[0:split_at]]
#val_x ,val_y = X1[rand[split_at:]],Y1[rand[split_at:]]


#f = h5py.File('data_ecssd_96x96_T.h5','w') 
f = h5py.File('data_merge_224x224x2_T.h5','w') 
#f = h5py.File('data_msra_192x192x2_T.h5','w') 
f['x'] = X1      
f['y'] = Y1
f['val_x'] = VAL_X
f['val_y'] = VAL_Y
f.close()                

#data_to_save = (X1, Y1)
#f = file('data_msra_200x150_T.cPickle', 'wb')
#pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
#f.close()
