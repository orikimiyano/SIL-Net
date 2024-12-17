from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras import backend as K
import tensorflow as tf

label_1 = [0, 0, 0]

label_2 = [128, 0, 0]
label_3 = [0, 128, 0]
label_4 = [128, 128, 0]
label_5 = [0, 0, 128]
label_6 = [128, 0, 128]
label_7 = [0, 128, 128]
label_8 = [128, 128, 128]
label_9 = [64, 0, 0]
label_10 = [192, 0, 0]
label_11 = [64, 128, 0]

label_12 = [192, 128, 0]
label_13 = [64, 0, 128]
label_14 = [192, 0, 128]
label_15 = [64, 128, 128]
label_16 = [192, 128, 128]
label_17 = [0, 64, 0]
label_18 = [128, 64, 0]
label_19 = [0, 192, 0]
label_20 = [128, 192, 0]

#COLOR_DICT = np.array([label_1, label_2, label_12, label_8])


COLOR_DICT = np.array(
    [label_1, label_2, label_2, label_4, label_12, label_6, label_8, label_8, label_9, label_10, label_11, label_2,
     label_12, label_14, label_15, label_16, label_17, label_18, label_19, label_20])

#COLOR_DICT = np.array([label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9, label_10, label_11, label_12, label_13, label_14, label_15, label_16, label_17, label_18, label_19, label_20])

IMAGE_SIZE = 256


def adjustData(img_1, img_2, img_3, img_B, label_A, label_B, flag_multi_class, num_class):
    if flag_multi_class:
        img_1 = img_1 / 255.
        img_2 = img_2 / 255.
        img_3 = img_3 / 255.
        img_B = img_B / 255.
        label_A = label_A[:, :, :, 0] if (len(label_A.shape) == 4) else label_A[:, :, 0]
        new_label_A = np.zeros(label_A.shape + (num_class,))
        label_B = label_B[:, :, :, 0] if (len(label_B.shape) == 4) else label_B[:, :, 0]
        new_label_B = np.zeros(label_B.shape + (num_class,))
        for i in range(num_class):
            new_label_A[label_A == i, i] = 1
        label_A = new_label_A
        for i in range(num_class):
            new_label_B[label_B == i, i] = 1
        label_B = new_label_B
    elif (np.max(img_1) > 1):
        img_1 = img_1 / 255.
        img_2 = img_2 / 255.
        img_3 = img_3 / 255.
        img_B = img_B / 255.
        label_A = label_A / 255.
        label_A[label_A > 0.5] = 1
        label_A[label_A <= 0.5] = 0
        label_B = label_B / 255.
        label_B[label_B > 0.5] = 1
        label_B[label_B <= 0.5] = 0
    return img_1, img_2, img_3, img_B, label_A, label_B


def trainGenerator(batch_size, aug_dict, train_path, image_folder_1, image_folder_2, image_folder_3, image_folder_B,
                   label_folder_A, label_folder_B, image_color_mode='grayscale',
                   label_color_mode='grayscale', image_save_prefix='image', label_save_prefix='label',
                   flag_multi_class=True, num_class=20, save_to_dir=None, target_size=(IMAGE_SIZE, IMAGE_SIZE), seed=1):
    image_1_datagen = ImageDataGenerator(**aug_dict)
    image_2_datagen = ImageDataGenerator(**aug_dict)
    image_3_datagen = ImageDataGenerator(**aug_dict)
    image_B_datagen = ImageDataGenerator(**aug_dict)
    label_A_datagen = ImageDataGenerator(**aug_dict)
    label_B_datagen = ImageDataGenerator(**aug_dict)

    image_1_generator = image_1_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_1],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    image_2_generator = image_2_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_2],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    image_3_generator = image_3_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_3],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    image_B_generator = image_B_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_B],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    label_A_generator = label_A_datagen.flow_from_directory(
        train_path,
        classes=[label_folder_A],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed
    )
    label_B_generator = label_B_datagen.flow_from_directory(
        train_path,
        classes=[label_folder_B],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed
    )
    train_generator = zip(image_1_generator, image_2_generator, image_3_generator, image_B_generator, label_A_generator,
                          label_B_generator)
    for img_1, img_2, img_3, img_B, label_A, label_B in train_generator:
        img_1, img_2, img_3, img_B, label_A, label_B = adjustData(img_1, img_2, img_3, img_B, label_A, label_B,
                                                                  flag_multi_class,
                                                                  num_class)
        yield [img_1, img_2, img_3, img_B], [label_A, label_B]


def getFileNum(test_path):
    for root, dirs, files in os.walk(test_path):
        lens = len(files)
        return lens


def testGenerator(test_path_1, test_path_2, test_path_3, test_path_B, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                    flag_multi_class=True,
                    as_gray=True):
    num_image_A = getFileNum(test_path_1)
    num_image_B = getFileNum(test_path_B)
    for i in range(num_image_A):
        img_1 = io.imread(os.path.join(test_path_1, "%d.png" % i), as_gray=as_gray)
        img_1 = img_1 / 255
        img_1 = trans.resize(img_1, target_size)
        img_1 = np.reshape(img_1, img_1.shape + (1,)) if (not flag_multi_class) else img_1
        img_1 = np.reshape(img_1, (1,) + img_1.shape)

        img_2 = io.imread(os.path.join(test_path_2, "%d.png" % i), as_gray=as_gray)
        img_2 = img_2 / 255
        img_2 = trans.resize(img_2, target_size)
        img_2 = np.reshape(img_2, img_2.shape + (1,)) if (not flag_multi_class) else img_2
        img_2 = np.reshape(img_2, (1,) + img_2.shape)

        img_3 = io.imread(os.path.join(test_path_3, "%d.png" % i), as_gray=as_gray)
        img_3 = img_3 / 255
        img_3 = trans.resize(img_3, target_size)
        img_3 = np.reshape(img_3, img_3.shape + (1,)) if (not flag_multi_class) else img_3
        img_3 = np.reshape(img_3, (1,) + img_3.shape)

        img_B = io.imread(os.path.join(test_path_B, "%d.png" % i), as_gray=as_gray)
        img_B = img_B / 255
        img_B = trans.resize(img_B, target_size)
        img_B = np.reshape(img_B, img_B.shape + (1,)) if (not flag_multi_class) else img_B
        img_B = np.reshape(img_B, (1,) + img_B.shape)

        yield [img_1, img_2, img_3, img_B]


# def testGenerator_B(test_path_1, test_path_2, test_path_3, test_path_B, target_size=(IMAGE_SIZE, IMAGE_SIZE), flag_multi_class=True, as_gray=True):
#     num_image = getFileNum(test_path_B)
#     for i in range(num_image):
#         img_1 = io.imread(os.path.join(test_path_1, "%d.png" % i), as_gray=as_gray)
#         img_1 = img_1 / 255
#         img_1 = trans.resize(img_1, target_size)
#         img_1 = np.reshape(img_1, img_1.shape + (1,)) if (not flag_multi_class) else img_1
#         img_1 = np.reshape(img_1, (1,) + img_1.shape)
#
#         img_2 = io.imread(os.path.join(test_path_2, "%d.png" % i), as_gray=as_gray)
#         img_2 = img_2 / 255
#         img_2 = trans.resize(img_2, target_size)
#         img_2 = np.reshape(img_2, img_2.shape + (1,)) if (not flag_multi_class) else img_2
#         img_2 = np.reshape(img_2, (1,) + img_2.shape)
#
#         img_3 = io.imread(os.path.join(test_path_3, "%d.png" % i), as_gray=as_gray)
#         img_3 = img_3 / 255
#         img_3 = trans.resize(img_3, target_size)
#         img_3 = np.reshape(img_3, img_3.shape + (1,)) if (not flag_multi_class) else img_3
#         img_3 = np.reshape(img_3, (1,) + img_3.shape)
#         img_B = io.imread(os.path.join(test_path_B, "%d.png" % i), as_gray=as_gray)
#         img_B = img_B / 255
#         img_B = trans.resize(img_B, target_size)
#         img_B = np.reshape(img_B, img_B.shape + (1,)) if (not flag_multi_class) else img_B
#         img_B = np.reshape(img_B, (1,) + img_B.shape)
#
#         yield [img_1, img_2, img_3, img_B]


def saveResult(save_path_A, save_path_B, npyfile_A, flag_multi_class=True):
    for i, item in enumerate(npyfile_A):
        if flag_multi_class:
            for slice in range(item.shape[0]):
                img = item[slice,]
                img_out = np.zeros(img[:, :, 0].shape + (3,))
                for row in range(img.shape[0]):
                    for col in range(img.shape[1]):
                        index_of_class = np.argmax(img[row, col])
                        img_out[row, col] = COLOR_DICT[index_of_class]
                        # img_out[row, col] = index_of_class
                img = img_out.astype(np.uint8)
                if i == 0:
                    io.imsave(os.path.join(save_path_A, str(slice) + '_predict.png'), img)
                elif i == 1:
                    io.imsave(os.path.join(save_path_B, str(slice) + '_predict.png'), img)
        else:
            for slice in range(item.shape[0]):
                img = item[slice,]
                img = img[:, :, 0]
                img[img > 0.5] = 1
                img[img <= 0.5] = 0
                img = img * 255.
                if i == 0:
                    io.imsave(os.path.join(save_path_A, str(slice) + '_predict.png'), img)
                elif i == 1:
                    io.imsave(os.path.join(save_path_B, str(slice) + '_predict.png'), img)
