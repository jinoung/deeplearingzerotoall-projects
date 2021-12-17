import torch
print('pytorch version: {}'.format(torch.__version__))

import os
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import check_util.checker as checker

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

batch_size = 100
num_epochs = 30
learning_rate = 0.00003

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

class Dataset(Dataset):
    def __init__(self, data_dir, mode, mean=None, std=None, seq_len=480, target_delay=24, stride=5, normalize=True):
        self.mode = mode
        self.seq_len = seq_len
        self.target_delay = target_delay
        self.stride = stride
        all_files = sorted(glob.glob(os.path.join(data_dir, mode, '*')))
        self.data = preprocess(all_files)
        if mode == 'train':
            assert (mean is None) and (std is None), \
                "평균과 분산은 train 폴더의 있는 데이터로 구하기 때문에 None 으로 설정합니다."
            ## 코드 시작 ##
            self.mean = np.mean(self.data, axis=0)
            self.std = np.std(self.data, axis=0)
            ## 코드 종료 ##
        else:
            assert (mean is not None) and (std is not None), \
                "평균과 분산은 `train_data`변수에 내장한 self.mean 과 self.std 를 사용합니다."
            ## 코드 시작 ##
            self.mean = mean
            self.std = std
            ## 코드 종료 ##

        if normalize:
            self.data = (self.data - self.mean) / self.std

    def __getitem__(self, index):
        ## 코드 시작 ##
        data_index = index * self.stride
        sequence = self.data[data_index:(data_index + self.seq_len)]
        target = self.data[data_index + self.seq_len + self.target_delay - 1][0]
        target = np.expand_dims(target, axis=0)
        ## 코드 종료 ##
        return sequence, target

    def __len__(self):
        max_idx = len(self.data) - self.seq_len - self.target_delay
        num_of_idx = max_idx // self.stride
        return num_of_idx

data_dir = './data/climate_seoul'
train_data = Dataset(data_dir, 'train', mean=None, std=None)
val_data = Dataset(data_dir, 'val', mean=train_data.mean, std=train_data.std)
test_data = Dataset(data_dir, 'test', mean=train_data.mean, std=train_data.std)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

#check 1
checker.customized_dataset_check(train_data)

#데이터 샘플 시각화
# temp = train_data[0][0]
# temp = temp[:, 0]
# plt.plot(range(len(temp)), temp)
# plt.xlabel('time')
# plt.ylabel('temperature\n(normalized)')
# plt.show()

def eval_baseline(data_loader, criterion):
    total_loss = 0
    cnt = 0
    for step, (sequence, target) in enumerate(data_loader):
        ## 코드 시작 ##
        # print(sequence.shape, target.shape)
        # sequence.torch.Size([100, 480, 9]) target.torch.Size([100, 1])
        pred = torch.empty(batch_size, 1)  # (100,1) 배열
        for i in range(batch_size):
            pred[i][0] = sequence[i][-1][0]
        #print(pred.shape, target.shape)
        loss = criterion(pred, target)
        ## 코드 종료 ##
        total_loss += loss
        cnt += 1
    avrg_loss = total_loss / cnt
    print('Baseline Average Loss: {:.4f}'.format(avrg_loss))
    return avrg_loss.item()

baseline_loss = eval_baseline(test_loader, nn.MSELoss())

for i in range(15):
    data_idx = np.random.randint(len(test_data))
    pred = test_data[data_idx][0][-1, 0]
    pred = pred * test_data.std[0] + test_data.mean[0]  # 예측 기온을 normalization 이전 상태(섭씨 단위)로 되돌리는 작업
    target = test_data[data_idx][1][0] * test_data.std[0] + test_data.mean[0]  # 실제 기온을 normalization 이전 상태(섭씨 단위)로 되돌리는 작업
    print('예측 기온: {:.1f} / 실제 기온: {:.1f}'.format(pred, target))


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        ## 코드 시작 ##
        self.lstm = nn.LSTM(input_size=9, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        ## 코드 종료 ##

    def init_hidden(self, batch_size):
        # 코드 시작
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # 위의 설명 3. 을 참고하여 None을 채우세요.
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)  # 위의 설명 3. 을 참고하여 None을 채우세요.
        # 코드 종료
        return hidden, cell

    def forward(self, x):
        # hidden, cell state init
        h, c = self.init_hidden(x.size(0))
        h, c = h.to(x.device), c.to(x.device)
        ## 코드 시작 ##
        # print('x.shape=',x.shape) #[100, 420, 9] == batch, seq, input
        out, (h, c) = self.lstm(x, (h, c))  # 위의 설명 4. 를 참고하여 None을 채우세요.
        # print('out.shape=', out.shape) #[100, 420, 100] == batch, seq, hidden --> batch, feature로 바꿔야
        fc_in = out[:, -1, :]
        final_output = self.fc(fc_in)  # final_output이 [100, 1] 이 되어야함. 위의 설명 5. 를 참고하여 None을 채우세요.
        ## 코드 종료 ##
        return final_output


#checker 2
checker.model_check(SimpleLSTM(), batch_size)

def train(num_epochs, model, data_loader, criterion, optimizer, saved_dir, val_every, device):
    print('Start training..')
    best_loss = 9999999
    for epoch in range(num_epochs):
        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)
            ## 코드 시작 ##
            outputs = model(sequence)  # 위의 설명 1. 을 참고하여 None을 채우세요.
            loss = criterion(outputs, target)     # 위의 설명 2. 를 참고하여 None을 채우세요.

            optimizer.zero_grad()            # Clear gradients: 위의 설명 3. 을 참고하여 None을 채우세요.
            loss.backward()            # Gradients 계산: 위의 설명 3. 을 참고하여 None을 채우세요.
            optimizer.step()            # Parameters 업데이트: 위의 설명 3. 을 참고하여 None을 채우세요.
            ## 코드 종료 ##

            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item()))

        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir)


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)
            ## 코드 시작 ##
            outputs = model(sequence)
            loss = criterion(outputs, target)
            ## 코드 종료 ##
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}'.format(epoch, avrg_loss))
    model.train()
    return avrg_loss


def test(model, data_loader, criterion, baseline_loss, device):
    print('Start test..')
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)
            ## 코드 시작 ##
            outputs = model(sequence)
            loss = criterion(outputs, target)
            ## 코드 종료 ##
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
        print('Test  Average Loss: {:.4f}  Baseline Loss: {:.4f}'.format(avrg_loss, baseline_loss))

    if avrg_loss < baseline_loss:
        print('베이스라인 성능을 뛰어 넘었습니다!')
    else:
        print('아쉽지만 베이스라인 성능을 넘지 못했습니다.')

def save_model(model, saved_dir, file_name='best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    output_path = os.path.join(saved_dir, file_name)
    ## 코드 시작 ##
    torch.save(check_point, output_path)
    ## 코드 종료 ##


torch.manual_seed(7777)  # 일관된 weight initialization을 위한 random seed 설정
## 코드 시작 ##
model = SimpleLSTM()  # 위의 설명 1. 을 참고하여 None을 채우세요.
model = model.to(device)
criterion = torch.nn.MSELoss().to(device)  # 위의 설명 2. 를 참고하여 None을 채우세요.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 위의 설명 3. 을 참고하여 None을 채우세요.
## 코드 종료 ##
val_every = 1
saved_dir = './saved/LSTM'

#checker 3
checker.loss_func_check(criterion)
checker.optim_check(optimizer)

train(num_epochs, model, train_loader, criterion, optimizer, saved_dir, val_every, device)

model_path = './saved/LSTM/best_model.pt'
# model_path = './saved/pretrained/LSTM/best_model.pt' # 모델 학습을 끝까지 진행하지 않은 경우에 사용
model = SimpleLSTM().to(device) # 아래의 모델 불러오기를 정확히 구현했는지 확인하기 위해 새로 모델을 선언하여 학습 이전 상태로 초기화

## 코드 시작 ##
checkpoint = torch.load(model_path)    # 위의 설명 1. 을 참고하여 None을 채우세요.
state_dict = checkpoint['net']    # 위의 설명 2. 를 참고하여 None을 채우세요.
model.load_state_dict(state_dict)                 # 위의 설명 3. 을 참고하여 None을 채우세요.
## 코드 종료 ##

test(model, test_loader, criterion, baseline_loss, device)

for i in range(15):
    data_idx = np.random.randint(len(test_data))
    sequence = test_data[data_idx][0]
    sequence = torch.Tensor(sequence).unsqueeze(0).to(device)

    pred = model(sequence)
    pred = pred.item() * test_data.std[0] + test_data.mean[0]
    target = test_data[data_idx][1][0] * test_data.std[0] + test_data.mean[0]
    print('예측 기온: {:.1f} / 실제 기온: {:.1f}'.format(pred, target))
