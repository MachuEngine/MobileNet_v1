### Git Repository 

구현 코드 및 Training된 모델 파일은 본 레포지토리에 포함되어 있습니다. 
```bash
git clone https://github.com/MachuEngine/MobileNet_v1.git
```

### Blog 
아래 Velog에 모델링 구현 및 추론 과정에 대해 조금 더 세부적인 내용을 작성해두었습니다. 
```
https://velog.io/@machu8/MobileNetv1
```

---

### Training 과정

```
Using device: cpu
Files already downloaded and verified
Files already downloaded and verified
Epoch 1: Train Loss: 1.5678, Train Acc: 0.4188
Epoch 1: Val Loss: 1.2877, Val Acc: 0.5402
Saved best model with accuracy: 0.5402
Epoch 2: Train Loss: 1.1023, Train Acc: 0.6079
Epoch 2: Val Loss: 0.9359, Val Acc: 0.6697
Saved best model with accuracy: 0.6697
Epoch 3: Train Loss: 0.8548, Train Acc: 0.6983
Epoch 3: Val Loss: 0.7417, Val Acc: 0.7369
Saved best model with accuracy: 0.7369
Epoch 4: Train Loss: 0.7007, Train Acc: 0.7575
Epoch 4: Val Loss: 0.6299, Val Acc: 0.7797
Saved best model with accuracy: 0.7797
Epoch 5: Train Loss: 0.5887, Train Acc: 0.7943
Epoch 5: Val Loss: 0.5754, Val Acc: 0.8046
Saved best model with accuracy: 0.8046
Epoch 6: Train Loss: 0.5208, Train Acc: 0.8189
Epoch 6: Val Loss: 0.5486, Val Acc: 0.8147
Saved best model with accuracy: 0.8147
Epoch 7: Train Loss: 0.4687, Train Acc: 0.8382
Epoch 7: Val Loss: 0.4860, Val Acc: 0.8331
Saved best model with accuracy: 0.8331
Epoch 8: Train Loss: 0.4197, Train Acc: 0.8555
Epoch 8: Val Loss: 0.5105, Val Acc: 0.8248
Epoch 9: Train Loss: 0.3838, Train Acc: 0.8688
Epoch 9: Val Loss: 0.4377, Val Acc: 0.8497
Saved best model with accuracy: 0.8497
Epoch 10: Train Loss: 0.3484, Train Acc: 0.8784
Epoch 10: Val Loss: 0.4534, Val Acc: 0.8448
Training complete. Best validation accuracy: 0.8497
```


---

### Predict 과정

#### Input image
![Input](./data/deer_picture.jpg)

#### CIFAR-10 데이터셋의 클래스 

| 라벨(Index) | 클래스 이름(Class Name) |
|:----------:|:-----------------------:|
| 0          | airplane               |
| 1          | automobile             |
| 2          | bird                   |
| 3          | cat                    |
| 4          | deer                   |
| 5          | dog                    |
| 6          | frog                   |
| 7          | horse                  |
| 8          | ship                   |
| 9          | truck                  |



<br>

#### 예측 결과: 확률 기반 top 5
1-best가 클래스 4인 사슴으로 예측 확인
```
Using device: cpu
Top 5 예측 결과:
클래스 4: 확률 0.9937
클래스 2: 확률 0.0058
클래스 5: 확률 0.0003
클래스 0: 확률 0.0001
클래스 3: 확률 0.0001
```

---

### Reference
MobileNets_Efficient Convolutional Neural Networks for Mobile Vision Applications (2018)