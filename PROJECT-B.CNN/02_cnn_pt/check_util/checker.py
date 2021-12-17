import torch
import torch.nn as nn
import os
import glob
import codecs
import csv


file_path = './check_util/cnn_submission.tsv'
lines = []
with codecs.open(file_path, 'r', encoding='utf-8', errors='replace') as fdata:
    rdr = csv.reader(fdata, delimiter="\t")
    for line in rdr:
        lines.append(line)

def submission_csv_write(writer, lines, fix_line_idx, flag):
    for i, line in enumerate(lines):
        new_line = lines[i]
        if i == fix_line_idx:
            new_line[3] = 'Pass' if flag else 'Fail'
        writer.writerow(new_line)


def dataset_check(output_dir):
    try:
        all_train_cat = glob.glob(os.path.join(output_dir, 'train', 'cat', '*'))
        all_train_dog = glob.glob(os.path.join(output_dir, 'train', 'dog', '*'))
        all_val_cat = glob.glob(os.path.join(output_dir, 'val', 'cat', '*'))
        all_val_dog = glob.glob(os.path.join(output_dir, 'val', 'dog', '*'))
        all_test_cat = glob.glob(os.path.join(output_dir, 'test', 'cat', '*'))
        all_test_dog = glob.glob(os.path.join(output_dir, 'test', 'dog', '*'))

        print('훈련용 고양이 이미지 개수:', len(all_train_cat))
        print('훈련용 강아지 이미지 개수:', len(all_train_dog))
        print('검증용 고양이 이미지 개수:', len(all_val_cat))
        print('검증용 강아지 이미지 개수:', len(all_val_dog))
        print('테스트용 고양이 이미지 개수:', len(all_test_cat))
        print('테스트용 강아지 이미지 개수:', len(all_test_dog))
        
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 1, True)
        
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def customized_dataset_check(dataset):
    item_flag = True
    len_flag = True

    try:
        shape = dataset[0][0].shape
        if shape != torch.Size([3, 120, 120]):
            print(f'{dataset.mode}의 __getitem__ 함수가 반환하는 img의 크기가 올바르지 않습니다. transform 과정에서 이미지 크기를 120으로 고정하는 부분을 실수로 수정했는지 확인하시기 바랍니다.')
            item_flag = False

    except AttributeError:
        print(f'{dataset.mode}의 __getitem__ 함수가 제대로된 img를 반환하지 않거나 transform이 img에 적용되지 않았습니다. __getitem__함수에서 transform을 적용하도록 구현했는지 확인하시기 바랍니다.')
        item_flag = False

    try:
        if len(dataset) != 2000:
            print('{}의 __len__ 함수가 train_data의 데이터 갯수를 2000이 아닌 {}으로 반환하고 있습니다. 가지고 있는 데이터셋 또는 __len__함수 구현에 문제가 없는지 다시 확인하시기 바랍니다.'.format(dataset.mode, len(dataset)))
            len_flag = False

        label = dataset[0][1]
        if label != 0:
            print(f'{dataset.mode}의 __getitem__이 반환하는 label이 올바르지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            item_flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 2, item_flag)

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 3, len_flag)

        if item_flag and len_flag:
            print('CatDogDataset class를 잘 구현하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def model_check(model):
    conv_flag = True
    bn_flag = True
    relu_flag = True
    mp_flag = True
    fc_flag = True
    all_in_channels = []
    all_out_channels = []
    all_kernel_size = []
    all_bn_num_feat = []
    all_maxpool_kernel_size = []
    num_relu = 0
    all_in_feat = []
    all_out_feat = []
    target_in_channels = [3, 32, 64, 128]
    target_out_channels = [32, 64, 128, 128]
    target_kernel_size = (3, 3)
    target_maxpool_kernel_size = (2, 2)
    target_in_feat = [3200, 512]
    target_out_feat = [512, 2]
    try:
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                for l in layer:
                    if isinstance(l, nn.Conv2d):
                        all_in_channels.append(l.in_channels)
                        all_out_channels.append(l.out_channels)
                        all_kernel_size.append(l.kernel_size)
                    elif isinstance(l, nn.BatchNorm2d):
                        all_bn_num_feat.append(l.num_features)
                    elif isinstance(l, nn.ReLU):
                        num_relu += 1
                    elif isinstance(l, nn.MaxPool2d):
                        all_maxpool_kernel_size.append(l.kernel_size)
            if isinstance(layer, nn.Linear):
                all_in_feat.append(layer.in_features)
                all_out_feat.append(layer.out_features)

        if len(all_in_channels) != 4:
            print('지문의 지시보다 더 많거나 적은 convolution layer가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            return

        if len(all_bn_num_feat) != 4:
            print('지문의 지시보다 더 많거나 적은 batch normalization layer가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            return

        if num_relu != 4:
            print('지문의 지시보다 더 많거나 적은 ReLU 함수가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            relu_flag = False

        if len(all_maxpool_kernel_size) != 4:
            print('지문의 지시보더 더 많거나 적은 maxpooling layer가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            return

        if len(all_in_feat) != 2:
            print('지문의 지시보다 더 많거나 적은 FC layer가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            return

        for i, (in_channels, target) in enumerate(zip(all_in_channels, target_in_channels)):
            if in_channels != target:
                print('{}번째 convolution layer의 입력 채널 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                conv_flag = False

        for i, (out_channels, target) in enumerate(zip(all_out_channels, target_out_channels)):
            if out_channels != target:
                print('{}번째 convolution layer의 출력 채널 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                conv_flag = False

        for i, k_size in enumerate(all_kernel_size):
            if k_size != target_kernel_size:
                print('{}번째 convolution layer의 필터 크기가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                conv_flag = False

        for i, (num_feat, target) in enumerate(zip(all_bn_num_feat, target_out_channels)):
            if num_feat != target:
                print('{}번째 batch normalization layer의 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                bn_flag = False

        for i, k_size in enumerate(all_maxpool_kernel_size):
            if k_size != target_maxpool_kernel_size and (k_size, k_size) != target_maxpool_kernel_size:
                print('{}번째 maxpooling layer의 필터 크기가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                mp_flag = False

        for i, (in_feat, target) in enumerate(zip(all_in_feat, target_in_feat)):
            if in_feat != target:
                print('{}번째 FC layer의 입력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                fc_flag = False

        for i, (out_feat, target) in enumerate(zip(all_out_feat, target_out_feat)):
            if out_feat != target:
                print('{}번째 FC layer의 출력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i + 1))
                fc_flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 4, conv_flag)

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 5, bn_flag)

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 6, relu_flag)

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 7, mp_flag)

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 8, fc_flag)

        if conv_flag and bn_flag and relu_flag and mp_flag and fc_flag:
            print('네트워크를 잘 구현하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def loss_func_check(criterion):
    flag = True
    try:
        if not isinstance(criterion, nn.CrossEntropyLoss):
            print('Cross entropy loss function이 정의되지 않았습니다. 지문을 다시 확인하시기 바랍니다.')
            flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 9, flag)

        if flag:
            print('Cross entropy loss function을 잘 정의하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def optim_check(optim):
    flag = True
    try:
        if not isinstance(optim, torch.optim.Adam):
            print('Adam optimizer가 정의되지 않았습니다. 지문을 다시 확인하시기 바랍니다.')
            flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 10, flag)

        if flag:
            print('Adam optimizer를 잘 정의하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def final_fc_check(new_model):
    flag = True
    try:
        in_feat = new_model.fc.in_features
        out_feat = new_model.fc.out_features

        if in_feat != 2048:
            print('마지막 FC layer의 입력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.')
            flag = False
        if out_feat != 2:
            print('마지막 FC layer의 출력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.')
            flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 11, flag)

        if flag:
            print('마지막 FC layer 잘 수정하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)