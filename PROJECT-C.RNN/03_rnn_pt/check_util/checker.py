import torch
import torch.nn as nn
import codecs
import csv
import os
import glob
import numpy as np

file_path = './check_util/rnn_submission.tsv'
lines = []
with codecs.open(file_path, 'r', encoding='utf-8', errors='replace') as fdata:
    rdr = csv.reader(fdata, delimiter='\t')
    for line in rdr:
        lines.append(line)


def submission_csv_write(writer, lines, fix_line_idx, flag):
    for i, line in enumerate(lines):
        new_line = lines[i]
        if i == fix_line_idx:
            new_line[3] = 'Pass' if flag else 'Fail'
        writer.writerow(new_line)

def preprocess(all_files):
    data_0 = [] # 기온
    data_1 = [] # 강수량
    data_2 = [] # 풍속
    data_3 = [] # 습도
    data_4 = [] # 증기압
    data_5 = [] # 이슬점 온도
    data_6 = [] # 현지 기압
    data_7 = [] # 해면 기압
    data_8 = [] # 지면 온도
    for f in all_files:
        with open(f, encoding='euc-kr') as c:
            csv_reader = csv.reader(c, delimiter=',')
            header = True
            for col in csv_reader:
                if header:
                    header = False
                    continue
                data_0.append(float(col[2])) if col[2] != '' else data_0.append(0.0)
                data_1.append(float(col[3])) if col[3] != '' else data_1.append(0.0)
                data_2.append(float(col[4])) if col[4] != '' else data_2.append(0.0)
                data_3.append(float(col[6])) if col[6] != '' else data_3.append(0.0)
                data_4.append(float(col[7])) if col[7] != '' else data_4.append(0.0)
                data_5.append(float(col[8])) if col[8] != '' else data_5.append(0.0)
                data_6.append(float(col[9])) if col[9] != '' else data_6.append(0.0)
                data_7.append(float(col[10])) if col[10] != '' else data_7.append(0.0)
                data_8.append(float(col[22])) if col[22] != '' else data_8.append(0.0)

    data = np.zeros((len(data_0), 9))
    for i, d in enumerate(data):
        data[i, 0] = data_0[i]
        data[i, 1] = data_1[i]
        data[i, 2] = data_2[i]
        data[i, 3] = data_3[i]
        data[i, 4] = data_4[i]
        data[i, 5] = data_5[i]
        data[i, 6] = data_6[i]
        data[i, 7] = data_7[i]
        data[i, 8] = data_8[i]
    return data

def customized_dataset_check(dataset):
    try:
        flag = True
        real_data = preprocess(sorted(glob.glob(os.path.join('./data/climate_seoul', dataset.mode, '*'))))
        sequence_shape = dataset[0][0].shape
        if sequence_shape != torch.Size([dataset.seq_len, 9]):
            print(f'{dataset.mode} 데이터의 __getitem__ 함수가 반환하는 seqeuence의 shape이 올바르지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            flag = False
        target_shape = dataset[0][1].shape
        if target_shape != torch.Size([1]):
            print(f'{dataset.mode} 데이터의 __getitem__ 함수가 반환하는 target의 shape 올바르지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            flag = False
        # 임의 idx 선정
        idx = torch.randint(0, len(dataset), size=(1,)).item()  
        tocheck_target = round(dataset[idx][1][0] * dataset.std[0] + dataset.mean[0], 4)
        real_target = real_data[idx*dataset.stride + dataset.seq_len + dataset.target_delay - 1, 0]
        if tocheck_target != real_target:
            print(f'{dataset.mode} 데이터의 __getitem__ 함수가 반환하는 target이 입력 시퀀스의 마지막 시점으로부터 {dataset.target_delay}시간 후의 기온이 아닙니다. 인덱싱을 올바르게 했는지 다시 확인하시기 바랍니다.')
            print(f'현재 target(normalize 이전): {tocheck_target}, 목적 target(normalize 이전): {real_target}')
            flag = False
            
        st_idx = torch.randint(0, dataset.seq_len, size=(1,)).item() #st_idx : 0~480
        tocheck_input = round(dataset[idx][0][st_idx, 0] * dataset.std[0] + dataset.mean[0], 4)
        real_input = real_data[idx*dataset.stride + st_idx, 0]        
        if tocheck_input != real_input:
            print(f'{dataset.mode} 데이터의 __getitem__ 함수가 반환하는 sequence가 self.data로부터 올바르게 인덱싱되고 있지 않습니다. stride를 고려하여 시작 인덱스를 올바르게 구했는지 다시 확인하시기 바랍니다.')
            print(f'현재 인덱싱된 첫번째 줄의 기온 데이터(normalize 이전): {tocheck_input}, 목적 데이터(normalize 이전): {real_input}')
            flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 1, flag)

        if flag:
            print('Dataset class를 잘 구현하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def model_check(model, batch_size):
    lstm_flag = True
    fc_flag = True
    forward_flag = True
    hidden_flag = True
    try:
        if model.lstm.input_size != 9:
            print('LSTM의 input_size가 올바르지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            lstm_flag = False

        if model.fc.in_features != model.hidden_size:
            print('FC layer의 입력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.')
            fc_flag = False

        if model.fc.out_features != 1:
            print('FC layer의 입력 feature 수가 잘못되었습니다. 지문을 다시 확인하시기 바랍니다.')
            fc_flag = False

        h, c = model.init_hidden(batch_size)
        if h.shape != torch.Size([model.num_layers, batch_size, model.hidden_size]):
            print('init_hidden 함수가 반환하는 초기 hidden state의 shape이 올바르지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            hidden_flag = False

        if c.shape != torch.Size([model.num_layers, batch_size, model.hidden_size]):
            print('init_hidden 함수가 반환하는 초기 cell state의 shape이 올바르지 않습니다. 지문을 다시 확인하시기 바랍니다.')
            hidden_flag = False

        x = torch.zeros(100, 420, 9)
        out = model(x)
        out_shape = out.shape
        print('out_shape=',out_shape)
        if out_shape != torch.Size([100, 1]):
            print('모델의 최종 출력의 shape...이 올바르지 않습니다. forward 함수를 정확히 구현했는지 다시 확인하시기 바랍니다.')
            forward_flag = False

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 2, lstm_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 3, fc_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 4, forward_flag)
        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 5, hidden_flag)

        if lstm_flag and fc_flag and forward_flag and hidden_flag:
            print('네트워를 잘 구현하셨습니다! 이어서 진행하셔도 좋습니다.')

    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)


def loss_func_check(criterion):
    flag = True
    try:
        if not isinstance(criterion, nn.MSELoss):
            print('MSE loss function이 정의되지 않았습니다. 지문을 다시 확인하시기 바랍니다.')

        with codecs.open(file_path, 'w', encoding='utf-8', errors='replace') as f:
            wr = csv.writer(f, delimiter='\t')
            submission_csv_write(wr, lines, 6, flag)

        if flag:
            print('MSE loss function을 잘 정의하셨습니다! 이어서 진행하셔도 좋습니다.')
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
            submission_csv_write(wr, lines, 7, flag)

        if flag:
            print('Adam optimizer를 잘 정의하셨습니다! 이어서 진행하셔도 좋습니다.')
    except Exception as e:
        print('체크 함수를 실행하는 도중에 다음과 같은 문제가 발생했습니다. 코드 구현을 완료했는지 다시 검토하시기 바랍니다.')
        print(e)
        
        
