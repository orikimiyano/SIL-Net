from data import *
from models.SILNet import *
import sys
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras_flops import get_flops

NameOfModel = 'SIL-Net'

##############1ST#################
Cross_Validation = '1ST'
train_index = Cross_Validation + '\data/train'


class Logger(object):
    def __init__(self, filename=str(NameOfModel) + ".log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/' + str(NameOfModel) + '.txt')

aug_args = dict()

train_gene = trainGenerator(batch_size=1, aug_dict=aug_args, train_path=str(train_index),
                            image_folder_1='T2', image_folder_2='T1', image_folder_3='T2_FS', image_folder_B='T2_TRA',
                            label_folder_A='矢状位标签', label_folder_B='横截面标签',
                            image_color_mode='grayscale', label_color_mode='grayscale',
                            image_save_prefix='image', label_save_prefix='label',
                            flag_multi_class=True, save_to_dir=None)

model = net(num_class=20)

# flops = get_flops(model, batch_size=1)
# print("FLOPS:"+str(flops))

model_checkpoint = ModelCheckpoint('saved_models/' + str(NameOfModel) + '_' + str(Cross_Validation) + '.hdf5',
                                   monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])

##############2ND#################

Cross_Validation = '2ND'
train_index = Cross_Validation + '\data/train'


class Logger(object):
    def __init__(self, filename=str(NameOfModel) + ".log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/' + str(NameOfModel) + '.txt')

aug_args = dict()

train_gene = trainGenerator(batch_size=1, aug_dict=aug_args, train_path=str(train_index),
                            image_folder_1='T2', image_folder_2='T1', image_folder_3='T2_FS', image_folder_B='T2_TRA',
                            label_folder_A='矢状位标签', label_folder_B='横截面标签',
                            image_color_mode='grayscale', label_color_mode='grayscale',
                            image_save_prefix='image', label_save_prefix='label',
                            flag_multi_class=True, save_to_dir=None)

model = net(num_class=20)

model_checkpoint = ModelCheckpoint('saved_models/' + str(NameOfModel) + '_' + str(Cross_Validation) + '.hdf5',
                                   monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])

##############3RD#################

Cross_Validation = '3RD'
train_index = Cross_Validation + '\data/train'


class Logger(object):
    def __init__(self, filename=str(NameOfModel) + ".log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/' + str(NameOfModel) + '.txt')

aug_args = dict()

train_gene = trainGenerator(batch_size=1, aug_dict=aug_args, train_path=str(train_index),
                            image_folder_1='T2', image_folder_2='T1', image_folder_3='T2_FS', image_folder_B='T2_TRA',
                            label_folder_A='矢状位标签', label_folder_B='横截面标签',
                            image_color_mode='grayscale', label_color_mode='grayscale',
                            image_save_prefix='image', label_save_prefix='label',
                            flag_multi_class=True, save_to_dir=None)

model = net(num_class=20)

model_checkpoint = ModelCheckpoint('saved_models/' + str(NameOfModel) + '_' + str(Cross_Validation) + '.hdf5',
                                   monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])

##############4TH#################

Cross_Validation = '4TH'
train_index = Cross_Validation + '\data/train'


class Logger(object):
    def __init__(self, filename=str(NameOfModel) + ".log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/' + str(NameOfModel) + '.txt')

aug_args = dict()

train_gene = trainGenerator(batch_size=1, aug_dict=aug_args, train_path=str(train_index),
                            image_folder_1='T2', image_folder_2='T1', image_folder_3='T2_FS', image_folder_B='T2_TRA',
                            label_folder_A='矢状位标签', label_folder_B='横截面标签',
                            image_color_mode='grayscale', label_color_mode='grayscale',
                            image_save_prefix='image', label_save_prefix='label',
                            flag_multi_class=True, save_to_dir=None)

model = net(num_class=20)

model_checkpoint = ModelCheckpoint('saved_models/' + str(NameOfModel) + '_' + str(Cross_Validation) + '.hdf5',
                                   monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])

##############5TH#################

Cross_Validation = '5TH'
train_index = Cross_Validation + '\data/train'


class Logger(object):
    def __init__(self, filename=str(NameOfModel) + ".log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log/' + str(NameOfModel) + '.txt')

aug_args = dict()

train_gene = trainGenerator(batch_size=1, aug_dict=aug_args, train_path=str(train_index),
                            image_folder_1='T2', image_folder_2='T1', image_folder_3='T2_FS', image_folder_B='T2_TRA',
                            label_folder_A='矢状位标签', label_folder_B='横截面标签',
                            image_color_mode='grayscale', label_color_mode='grayscale',
                            image_save_prefix='image', label_save_prefix='label',
                            flag_multi_class=True, save_to_dir=None)

model = net(num_class=20)

model_checkpoint = ModelCheckpoint('saved_models/' + str(NameOfModel) + '_' + str(Cross_Validation) + '.hdf5',
                                   monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_gene,
                              steps_per_epoch=2000,
                              epochs=10,
                              verbose=1,
                              callbacks=[model_checkpoint])