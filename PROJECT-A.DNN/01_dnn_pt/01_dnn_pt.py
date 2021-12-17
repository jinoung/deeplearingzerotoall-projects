''' 프로젝트 개요
Fashion-MNIST는 MNIST와 동일한 크기의 의류 이미지데이터
학습데이터 : 60,000개, 테스트데이터 : 10,000개
데이터 분류는 10가지 : T-Shirts, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Bag, Ankle boot
10 종류의 의류와 관련된 이미지를 학습시키고 판별하는 모델 작성
다뤄야 할 데이터는 1x28x28 (채널 x 이미지높이 x 이미지너비)의 흑백 이미지
DNN(Deep Neural Network)의 입력으로 사용되기 위해서 1 × 28 × 28의 3차원은 784의 1차원 데이터로(1*28*28=784) 변환
784차원의 입력 데이터는 DNN을 통과하여 10차원의 의류 종류를 나타내는 출력으로 변환
'''
import torch
print('pytorch version: {}'.format(torch.__version__))

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import check_util.checker as checker
#%matplotlib inline

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

'''학습에 필요한 하이퍼파리미터의 값을 초기화'''
batch_size = 100
num_epochs = 5
learning_rate = 0.001

from torch.utils.data import DataLoader

root = './data'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_data = dset.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_data = dset.FashionMNIST(root=root, train=False, transform=transform, download=True)
## 코드 시작 ##
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) #test data는 데이터를 무작위로 샘플링할 필요가 없음
## 코드 종료 ##

# labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
# 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}
# columns = 5
# rows = 5
# fig = plt.figure(figsize=(8, 8))

# for i in range(1, columns * rows + 1):
#     data_idx = np.random.randint(len(train_data))
# img = train_data[data_idx][0][0, :, :].numpy()  # numpy()를 통해 torch Tensor를 numpy array로 변환
# label = labels_map[train_data[data_idx][1]]  # item()을 통해 torch Tensor를 숫자로 변환
#
# fig.add_subplot(rows, columns, i)
# plt.title(label)
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()

class DNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DNN, self).__init__() #super : 자식클라스에서 부모클라스의 내용을 사용하고 싶을 때 사용
        self.layer1 = nn.Sequential(
            ## 코드 시작 ##
            torch.nn.Linear(28 * 28, 512, bias=True),  # Linear_1 해당하는 층
            torch.nn.BatchNorm1d(512),  # BatchNorm_1 해당하는 층
            torch.nn.ReLU()  # ReLU_1 해당하는 층
            ## 코드 종료 ##
        )
        self.layer2 = nn.Sequential(
            ## 코드 시작 ##
            torch.nn.Linear(512, 10, bias=True)  # Linear_2 해당하는 층
            ## 코드 종료 ##
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        return x_out

def weights_init(m):
    if isinstance(m, nn.Linear): # 모델의 모든 MLP 레이어에 대해서
        nn.init.xavier_normal_(m.weight) # Weight를 xavier_normal로 초기화
        print(m.weight)

torch.manual_seed(7777) # 일관된 weight initialization을 위한 random seed 설정
model = DNN().to(device)
model.apply(weights_init) # 모델에 weight_init 함수를 적용하여 weight를 초기화

## 코드 시작 ##
criterion = nn.CrossEntropyLoss() #criterion 변수에 Classification에서 자주 사용되는 Cross Entropy Loss를 정의
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optimizer 변수에 Adam optimizer를 정의
## 코드 종료 ##

for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        ## 코드 시작 ##
        outputs = model(imgs) #모델에 imgs 데이터를 주고, 그 출력을 outputs 변수에 저장
        loss = criterion(outputs, labels) #모델의 outputs과 train_loader에서 제공된 labels를 통해 손실값을 구하고, 그 결과를 loss 변수에 저장

        optimizer.zero_grad() #이전에 계산된 gradient를 모두 clear
        loss.backward() #Gradient를 계산
        optimizer.step() #Optimizer를 통해 파라미터를 업데이트
        ## 코드 종료 ##

        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax).float().mean()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

'''마지막으로 학습된 모델의 성능을 테스트할 차례입니다.
model.eval()은 모델을 평가(evaluation) 모드로 설정하겠다는 의미입니다. 
평가 모드 가 필요한 이유는, batch normalization과 dropout이 training을 할 때와 test를 할 때 작동하는 방식이 다르기 때문입니다. 
평가 모드를 설정해주어야 test를 할 때 일관된 결과를 얻을 수 있습니다.
torch.no_grad()는 torch.Tensor의 requires_grad를 False로 만들어줍니다. 
Test 때는 backpropagation을 통해 gradient를 계산할 필요가 없기 때문에, 
Tensor의 requires_grad를 False로 바꿔줌을 통해 메모리를 낭비하지 않을 수 있습니다.
Test를 마친 이후에 training을 더 진행하길 원하면 model.train()을 통해 다시 training 모드로 설정을 해주면 됩니다.
'''
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, argmax = torch.max(outputs, 1)  # max()를 통해 최종 출력이 가장 높은 class 선택
        total += imgs.size(0)
        correct += (labels == argmax).sum().item()

    print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))

columns = 5
rows = 5
fig = plt.figure(figsize=(8, 8))

'''학습된 모델의 예측 결과를 시각화하면 다음과 같습니다. 
괄호안에 'O'이 있는 경우, 모델이 정확한 예측을 한 것이고 'X'가 있는 경우는 틀린 예측을 한 것입니다. 
틀린 경우에는 모델의 예측과 함께 실제 정답을 표기해두었습니다.
'''
model.eval()
for i in range(1, columns * rows + 1):
    data_idx = np.random.randint(len(test_data))
    input_img = test_data[data_idx][0].unsqueeze(dim=0).to(device)
    '''
    unsqueeze()를 통해 입력 이미지의 shape을 (1, 28, 28)에서 (1, 1, 28, 28)로 변환. 
    모델에 들어가는 입력 이미지의 shape은 (batch_size, channel, width, height) 되어야 함에 주의!
    '''
    output = model(input_img)
    _, argmax = torch.max(output, 1)
    pred = labels_map[argmax.item()]
    label = labels_map[test_data[data_idx][1]]

    fig.add_subplot(rows, columns, i)
    if pred == label:
        plt.title(pred + '(O)')
    else:
        plt.title(pred + '(X)' + ' / ' + label)
    plot_img = test_data[data_idx][0][0, :, :]
    plt.imshow(plot_img, cmap='gray')
    plt.axis('off')
model.train()
plt.show()

# - Multi layer perceptron을 설계할 수 있다.
# - 네트워크에 ReLU, Batch normalization를 적용할 수 있다.
# - DataLoader를 이용하여 데이터를 로드할 수 있다.
# - 원하는 방식으로 가중치를 초기화할 수 있다.
# - Loss function과 optimizer를 정의할 수 있다.
# - Loss를 측정하고 gradient를 계산해 모델 파라미터를 업데이트할 수 있다.
# - 학습한 모델의 성능을 test 할 수 있다.