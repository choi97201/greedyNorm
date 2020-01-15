import numpy as np
import os
import cv2
import random
import math
import keras

def make_segmap(pred, test_step, model, epoch):
    print(pred.shape)
    print(pred.max(), pred.min())
    print('probilities >>>>> ', pred[0,128,128,:])
    print('sum of probilities >>>>> ', pred[0,128,128,:].sum())
    result = np.argmax(pred, axis=-1)
    result = np.squeeze(result)
    print('result >>>>> ', result.shape)
    print('result >>>>> ', np.unique(result))
    cv2.imwrite('./test_img/{}/{}_test{}.png'.format(model, epoch, test_step), result*10)
    print('test{}.png complete'.format(test_step))
    print(result.max(), result.min())


def labeling(lb, window_size, classes):
    yy = np.zeros((window_size[0], window_size[1], classes), dtype=np.uint8)
    for i in range(classes):
        yy[:,:,i][np.where(lb == i)] = 1
    
    return yy


def scaling(img, lb, window_size, mode):
    if img.shape[0] >= img.shape[1]:
        L = img.shape[0]
        img_aug = np.zeros((L,L,3), dtype=float)
        lb_aug = np.zeros((L,L), dtype=int)
        start = np.int(np.floor((L - img.shape[1]) // 2))
        finish = np.int(start + img.shape[1] - 1)
        # print(start, finish)
        img_aug[:,start:finish+1,:] = img
        lb_aug[:,start:finish+1] = lb
    else:
        L = img.shape[1]
        img_aug = np.zeros((L,L,3), dtype=float)
        lb_aug = np.zeros((L,L), dtype=int)
        start = np.int(np.floor((L - img.shape[0]) // 2))
        finish = np.int(start + img.shape[0] - 1)
        img_aug[start:finish+1,:,:] = img
        lb_aug[start:finish+1,:] = lb

    scale = random.random() * 1.5 + 0.5
    if mode in ['train', 'val']:
        L = math.ceil(L * scale)

        img_aug = cv2.resize(img_aug, (L, L), interpolation=cv2.INTER_NEAREST)
        lb_aug = cv2.resize(lb_aug, (L, L), interpolation=cv2.INTER_NEAREST)
        # print(img_aug.shape, lb_aug.shape)

        if L < window_size:
            img_crop = np.zeros((window_size,window_size,3), dtype=int)
            img_crop[(window_size-L)//2+1:(window_size+L)//2+1,(window_size-L)//2+1:(window_size+L)//2+1,:] = img_aug
            lb_crop = np.zeros((window_size,window_size), dtype=int)
            lb_crop[(window_size-L)//2+1:(window_size+L)//2+1,(window_size-L)//2+1:(window_size+L)//2+1] = lb_aug
        elif L > window_size:
            img_crop = img_aug[(L-window_size)//2+1:(L+window_size)//2+1,(L-window_size)//2+1:(L+window_size)//2+1,:]
            lb_crop = lb_aug[(L-window_size)//2+1:(L+window_size)//2+1,(L-window_size)//2+1:(L+window_size)//2+1]
        else:
            img_crop = img_aug
            lb_crop = lb_aug
        
        if random.random() > 0.5:
            img_crop = cv2.flip(img_crop, 1)
            lb_crop = cv2.flip(lb_crop, 1)

    elif mode == 'test':
        img_crop = cv2.resize(img_aug, (window_size, window_size), interpolation=cv2.INTER_NEAREST)
        img_crop = img_crop.astype(np.uint8)
        lb_crop = cv2.resize(lb_aug, (window_size, window_size), interpolation=cv2.INTER_NEAREST)

    # print(img_crop.shape, lb_crop.shape)
    return img_crop, lb_crop


class DataGenerator(keras.utils.Sequence):
    def __init__(self, datatype, dataset, batch_size, window_size, mode, classes, shuffle=True):
        self.datatype = datatype
        self.dataset = dataset
        self.batch_size = batch_size
        self.window_size = window_size
        self.mode = mode
        self.classes = classes
        self.shuffle = shuffle

    def on_epoch_begin(self):
        if self.shuffle:
            random.shuffle(self.dataset)

    def __data_generation(self, dataset_temp):
        x_data = np.empty((self.batch_size, self.window_size, self.window_size, 3))
        y_data = np.empty((self.batch_size, self.window_size*self.window_size, self.classes))

        for i, data in enumerate(dataset_temp):
            if self.datatype == "voc":   
                img = cv2.cvtColor(cv2.imread('../dataset/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(data)), cv2.COLOR_BGR2RGB).astype('float32')
                lb = cv2.imread('../dataset/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassAugRaw/{}.png'.format(data), cv2.IMREAD_GRAYSCALE)
            elif self.datatype == "atr":
                img = cv2.cvtColor(cv2.imread('../dataset/ATR/humanparsing/JPEGImages/{}.jpg'.format(data)), cv2.COLOR_BGR2RGB).astype('float32')
                lb = cv2.imread('../dataset/ATR/humanparsing/SegmentationClassAug/{}.png'.format(data), cv2.IMREAD_GRAYSCALE)
            img /= 255
            lb[np.where(lb == 255)] = 0
            img, lb = scaling(img, lb, self.window_size, self.mode)
            lb = labeling(lb, lb.shape, self.classes)

            img = img[np.newaxis,...]
            lb = lb[np.newaxis,...]
            lb = np.reshape(lb, (-1, self.window_size*self.window_size, self.classes))

            x_data[i] = img
            y_data[i] = lb

        return x_data, y_data

    def __len__(self):
        return int(len(self.dataset)//self.batch_size)

    def __getitem__(self, index):
        dataset_temp = self.dataset[index*self.batch_size:(index+1)*self.batch_size]
        
        X, Y = self.__data_generation(dataset_temp)

        return X, Y

    
def Adversarial_data(datatype, dataset, batch_size, window_size, mode, classes):
    x_data = np.empty((batch_size, window_size, window_size, 3))
    mask = np.empty((batch_size, window_size, window_size, 3))
    y_data = np.empty((batch_size, window_size*window_size, classes))
    z_noise = np.random.randn(batch_size, window_size, window_size, 3)

    for i, data in enumerate(dataset):
        if datatype == "voc":   
            img = cv2.cvtColor(cv2.imread('./dataset/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(data)), cv2.COLOR_BGR2RGB).astype('float32')
            lb = cv2.imread('./dataset/VOCdevkit/VOC2012/SegmentationClassAugRaw/{}.png'.format(data), cv2.IMREAD_GRAYSCALE)
        elif datatype == "cityscapes":
            img = cv2.cvtColor(cv2.imread('./dataset/cityscapes/TrainVal_images/train_images/{}.jpg'.format(data)), cv2.COLOR_BGR2RGB).astype('float32')
            lb = cv2.imread('./dataset/cityscapes/TrainVal_parsing_annotations/train_segmentations/{}.png'.format(data), cv2.IMREAD_GRAYSCALE)
        elif datatype == "atr":
            img = cv2.cvtColor(cv2.imread('./dataset/ATR/humanparsing/JPEGImages/{}.jpg'.format(data)), cv2.COLOR_BGR2RGB).astype('float32')
            lb = cv2.imread('./dataset/ATR/humanparsing/SegmentationClassAug/{}.png'.format(data), cv2.IMREAD_GRAYSCALE)


        img /= 255
        lb[np.where(lb == 255)] = 0
        img, lb = scaling(img, lb, window_size, mode)
        mask[i] = np.dstack((lb, lb, lb))
        lb = labeling(lb, lb.shape, classes)

        # img = img[np.newaxis,...]
        # lb = lb[np.newaxis,...]
        lb = np.reshape(lb, (window_size*window_size, classes))

        x_data[i] = img
        y_data[i] = lb

    return x_data, mask, y_data, z_noise



# from keras.preprocessing.image import ImageDataGenerator


# def combineGenerator(gen1, gen2):
#     while True:
#         yield gen1.next(), gen2.next()


# def generator(data_dir, target_size, batch_size, seed, mode):
#     if mode == 'train':
#         image_args = dict(rescale=1./255,
#                           zoom_range=[0.5, 2.0],
#                         #   width_shift_range=0.1,
#                         #   height_shift_range=0.1,
#                           fill_mode='constant',
#                           horizontal_flip=True)
#         mask_args = dict(zoom_range=[0.5, 2.0],
#                         #  width_shift_range=0.1,
#                         #  height_shift_range=0.1,
#                          fill_mode='constant',
#                          horizontal_flip=True)
                        
#     else:
#         image_args = dict(rescale=1./255)
#         mask_args = dict(rescale=1./255)

#     image_datagen = ImageDataGenerator(**image_args)
#     mask_datagen = ImageDataGenerator(**mask_args)

#     image_generator = image_datagen.flow_from_directory(data_dir,
#                                                         target_size=(target_size[0], target_size[1]),
#                                                         batch_size=batch_size,
#                                                         class_mode=None,
#                                                         classes=[mode+'_image'],
#                                                         seed=seed,
#                                                         shuffle=True)
#     mask_generator = mask_datagen.flow_from_directory(data_dir,
#                                                       target_size=(target_size[0], target_size[1]),
#                                                       batch_size=batch_size,
#                                                       class_mode=None,
#                                                       classes=[mode+'_mask'],
#                                                       color_mode='grayscale',
#                                                       seed=seed,
#                                                       shuffle=True)

#     return combineGenerator(image_generator, mask_generator)
    # return zip(image_generator, mask_generator)

# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
# img = load_img()


if __name__ == '__main__':
    trainset = [line.rstrip() for line in open('./atr_train.txt', 'r')]
    valset = [line.rstrip() for line in open('./atr_val.txt', 'r')]
    for i in range(10):
        print('######### {} #########'.format(i))
        gen = Adversarial_data('atr', trainset[i*4:(i+1)*4], 4, 512, 'train', 21)
        print(gen[0].shape, gen[1].shape, gen[2].shape, gen[3].shape)
