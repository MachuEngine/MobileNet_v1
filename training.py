import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse

# --- 1. 모델 구성 요소: Depthwise Separable Convolution ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        # 각 채널별로 독립적인 3x3 합성곱
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.bn_depth = nn.BatchNorm2d(in_channels)
        # 채널 결합을 위한 1x1 합성곱
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.bn_point = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn_depth(self.depthwise(x)))
        x = self.relu(self.bn_point(self.pointwise(x)))
        return x

# --- 2. MobileNet 모델 정의 ---
# 여기서는 CIFAR-10 분류를 위해 num_classes를 10으로 설정합니다.
class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 초기 표준 합성곱 (입력: 3채널, 출력: 32채널)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise Separable Convolution 블록들을 순차적으로 구성
        self.dsconv = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            # 5번 반복: 512 채널, stride 1
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1)
        )
        # Global Average Pooling 후 분류 레이어 연결
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dsconv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 평탄화 (flatten)
        x = self.fc(x)
        return x

# --- 3. 학습 함수 (한 에포크 단위) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()  # 학습 모드 설정
    running_loss = 0.0
    total = 0
    correct = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# --- 4. 검증 함수 ---
def validate(model, dataloader, criterion, device, epoch):
    model.eval()  # 평가 모드 설정
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch}: Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# --- 5. Main 함수: 데이터 로딩, 학습 및 검증 루프 ---
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 데이터 전처리: CIFAR-10 이미지는 원래 32x32이지만, 모델 입력 크기 224로 Resize 함.
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010]) 

    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010])
    ])
    
    # CIFAR-10 데이터셋 로딩 (학습, 검증)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 모델, 손실 함수, 옵티마이저 정의
    model = MobileNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        _, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # 검증 정확도가 개선되면 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print("Saved best model with accuracy:", best_val_acc)
    
    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))

# --- 6. 스크립트 실행 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MobileNet on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=10, help="학습 에포크 수")
    parser.add_argument('--batch_size', type=int, default=64, help="배치 사이즈")
    parser.add_argument('--lr', type=float, default=0.01, help="학습률")
    parser.add_argument('--save_path', type=str, default='./models/best_model.pth', help="최고 성능 모델 저장 경로")
    
    args = parser.parse_args()
    main(args)
