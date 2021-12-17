import torch
import torch.nn as nn
import codecs
import csv


file_path = './check_util/dnn_submission.tsv'
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


def train_loader_check(train_loader, test_loader):
    try:
        dset_flag = True
        batch_size_flag = True
        shuffle_flag = True
        if not train_loader.dataset.train:
            print('train_loader의 dataset class가 잘못되었습니다. 학습용 dataset class를 인자로 전달했는지 확인하시기 바랍니다.')
            dset_flag = False

        if train_loader.batch_size != 100:
            print('train_loader의 미니배치 크기가 기존에 정의한 크기와 다릅니다. \"2.하이퍼파라미터 세팅\"에서 정의한 배치크기를 인자로 활용했는지 확인하시기 바랍니다.')
            batch_size_flag = False

        if not isinstance(train_loader.sampler, torch.utils.data.sampler.RandomSampler):
            print('train_loader의 suffle 인자가 지문의 지시대로 되어 있지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            shuffle_flag = False

        if test_loader.dataset.train:
            print('test_loader의 dataset class가 잘못되었습니다. 테스트용 dataset class를 인자로 전달했는지 확인하시기 바랍니다.')
            dset_flag = False

        if test_loader.batch_size != 100:
            print('test_loader의 미니배치 크기가 기존에 정의한 크기와 다릅니다. \"2.하이퍼파라미터 세팅\"에서 정의한 배치크기를 인자로 활용했는지 확인하시기 바랍니다.')
            batch_size_flag = False

        if not isinstance(test_loader.sampler, torch.utils.data.sampler.SequentialSampler):
            print('test_loader의 suffle 인자가 지문의 지시대로 되어 있지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            shuffle_flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 1, dset_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 2, batch_size_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 3, shuffle_flag)

        if dset_flag and batch_size_flag and shuffle_flag:
            print('train_loader와 test_loader를 잘 구현하셨습니다! 이어서 진행하셔도 좋습니다.')

    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)

def model_check(model):
    fc_flag = True
    bn_flag = True
    relu_flag = True
    all_in_feat = []
    target_in_feat = [784, 512]
    all_out_feat = []
    target_out_feat = [512, 10]
    all_bn_num_feat = []
    target_bn_num_feat = [512]
    num_relu = 0

    try:
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                for l in layer:
                    if isinstance(l, nn.Linear):
                        all_in_feat.append(l.in_features)
                        all_out_feat.append(l.out_features)
                    elif isinstance(l, nn.BatchNorm1d):
                        all_bn_num_feat.append(l.num_features)
                    elif isinstance(l, nn.ReLU):
                        num_relu += 1

        if len(all_in_feat) != 2 or len(all_out_feat) != 2:
            print('지문의 지시보다 더 많거나 적은 FC layer가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            return

        if len(all_bn_num_feat) != 1:
            print('지문의 지시보다 더 많거나 적은 Batch normalization layer가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            return

        if num_relu != 1:
            print('지문의 지시보다 더 많거나 적은 ReLU 함수가 설계되었습니다. 지문을 다시 확인하시기 바랍니다.')
            relu_flag = False

        for i, (in_feat, target) in enumerate(zip(all_in_feat, target_in_feat)):
            if in_feat != target:
                print('{}번째 FC layer의 입력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i+1))
                fc_flag = False

        for i, (out_feat, target) in enumerate(zip(all_out_feat, target_out_feat)):
            if out_feat != target:
                print('{}번째 FC layer의 출력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i+1))
                fc_flag = False

        for i, (bn_num_feat, target) in enumerate(zip(all_bn_num_feat, target_bn_num_feat)):
            if bn_num_feat != target:
                print('{}번째 Batch normalization layer의 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.'.format(i+1))
                bn_flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 4, fc_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 5, bn_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 6, relu_flag)

        if fc_flag and bn_flag and relu_flag:
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
            submission_csv_write(wr, lines, 7, flag)

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

        if optim.defaults['lr'] != 0.001:
            print('Optimizer의 learning rate가 \"2.하이퍼파라미터 세팅\"에서 정의한 것과 다릅니다.')
            flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 8, flag)

        if flag:
            print('Adam optimizer를 잘 정의하셨습니다! 이어서 진행하셔도 좋습니다.')

    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)