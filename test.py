import os
import warnings
from data import *
from models.SILNet import *
from keras_flops import get_flops

warnings.filterwarnings("ignore")

NameOfModel = 'SIL-Net'

##############1ST#################
Cross_Validation = '5TH'
train_index = Cross_Validation + '\data/'
results_path_A ='results_A/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'
results_path_B ='results_B/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'

def test(case_path_1, case_path_2, case_path_3, case_root_B,case_name):

    testGene_A = testGenerator(case_path_1, case_path_2, case_path_3, case_root_B)
    # testGene_B = testGenerator_B(case_root_B, case_root_B, case_root_B, case_root_B)
    lens_A = getFileNum(case_path_1)
    # lens_B = getFileNum(case_root_B)
    model = net(num_class=20)
    # flops = get_flops(model, batch_size=1)
    # print("FLOPS:"+str(flops))
    model.load_weights('saved_models/' + str(NameOfModel) + '_' + str(Cross_Validation) + '.hdf5')
    results_A = model.predict_generator(testGene_A, lens_A, max_queue_size=1, verbose=1)
    # results_B = model.predict_generator(testGene_A, lens_B, max_queue_size=1, verbose=1)
    if not (os.path.exists(results_path_A + case_name)):
        os.makedirs(results_path_A + case_name)
    if not (os.path.exists(results_path_B + case_name)):
        os.makedirs(results_path_B + case_name)
    saveResult(results_path_A + case_name, results_path_B + case_name, results_A)


case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'
case_path_B = str(train_index) + 'test/T2_TRA'

# test(case_path_1,case_path_2,case_path_3,case_path_3)

print('1ST test')

for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        one_case_root_B = os.path.join(case_path_B, dir)
        test(one_case_root_1, one_case_root_2, one_case_root_3, one_case_root_B, dir)


##############2ND#################
Cross_Validation = '2ND'
train_index =  Cross_Validation + '\data/'
results_path_A ='results_A/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'
results_path_B ='results_B/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'
case_path_B = str(train_index) + 'test/T2_TRA'

print('2nd test')

for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        one_case_root_B = os.path.join(case_path_B, dir)
        test(one_case_root_1, one_case_root_2, one_case_root_3, one_case_root_B, dir)


##############3RD#################
Cross_Validation = '3RD'
train_index = Cross_Validation + '\data/'
results_path_A ='results_A/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'
results_path_B ='results_B/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'
case_path_B = str(train_index) + 'test/T2_TRA'

print('3RD test')

for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        one_case_root_B = os.path.join(case_path_B, dir)
        test(one_case_root_1, one_case_root_2, one_case_root_3, one_case_root_B, dir)


##############4TH#################
Cross_Validation = '4TH'
train_index = Cross_Validation + '\data/'
results_path_A ='results_A/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'
results_path_B ='results_B/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'
case_path_B = str(train_index) + 'test/T2_TRA'

print('4TH test')

for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        one_case_root_B = os.path.join(case_path_B, dir)
        test(one_case_root_1, one_case_root_2, one_case_root_3, one_case_root_B, dir)


##############5TH#################
Cross_Validation = '5TH'
train_index = Cross_Validation + '\data/'
results_path_A ='results_A/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'
results_path_B ='results_B/' + str(NameOfModel) + '_' + str(Cross_Validation) + '/'

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'
case_path_B = str(train_index) + 'test/T2_TRA'

print('5TH test')

for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        one_case_root_B = os.path.join(case_path_B, dir)
        test(one_case_root_1, one_case_root_2, one_case_root_3, one_case_root_B, dir)