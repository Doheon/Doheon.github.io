---
title: "[코드구현] Sentence Classification - KoBERT"
toc: true
toc_sticky: true
date: 2021-07-27
categories: Code-Implementation NLP
---

한국어에 대해 pre-train 되어 있는 BERT 모델인 KoBERT를 이용하여 뉴스 데이터의 카테고리를 분류하는 task를 직접 구현해 보았다.

Code: <https://github.com/Doheon/NewsClassification-KoBERT>

&nbsp;



사용한 모델과 데이터셋의 출처는 아래와 같다.

KoBERT: <https://github.com/SKTBrain/KoBERT>

Dataset: <http://ling.snu.ac.kr/class/cl_under1801/FinalProject.htm>



데이터셋은 8개의 카테고리가 있고 각 카테고리당 200개의 뉴스 기사가 존재한다.

&nbsp;



## Import Module

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

import os


from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
```

먼저 필요한 module들을 설치하고 import해준다. (kobert 설치 방법은 github 주소 참고)



&nbsp;

```python
device = torch.device("cuda:0")
# device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model()
```

사용할 deivce를 설정하고 pretrain 된 bert 모델을 불러온다.

&nbsp;



## Load Data

```python
dataset_train = []
dataset_test = []

root = "newsData/"
list = os.listdir(root)
for cat in list:
    files = os.listdir(root + cat)
    for i,f in enumerate(files):
        fname = root + cat + "/" + f
        file = open(fname, "r")
        strings = file.read()
        if i<170:
            dataset_train.append([strings, cat])
        else:
            dataset_test.append([strings,cat])
        file.close()

print(len(dataset_train), len(dataset_test))
```

```
1360 240
```



파일을 읽어서 기사 내용과 카테고리를 하나의 데이터로 list를 만든다.

각 카테고리당 170개를 train set, 30개를 test set으로 사용했다.



&nbsp;



```python
dataset_train[0]
```

```
['동남아 담당\' 北 최희철 부상 베이징 도착…싱가포르행 주목\t최 부상, 행선지·방문 목적 질문에는 \'묵묵부답\'\n\n(베이징=연합뉴스) 김진방 특파원 = 북한이 북미 정상회담 무산 가능성까지 거론하며 강경한 태도를 보이는 가운데 동남아시아 외교를 담당하는 최희철 북한 외무성 부상이 19일 중국 베이징 서우두(首都) 공항에 모습을 드러냈다.\n\n최 부상은 이날 오전 평양발 고려항공 JS151편을 이용해 베이징 서우두 공항에 도착했다.\n\n최 부상은 최종 목적지를 묻는 취재진의 질문에 아무런 답변을 하지 않고, 북한 대사관 관계자들과 함께 공항을 빠져나갔다.\n\n북미 정상회담을 20여 일 앞둔 상황에서 동남아 외교통인 최 부상이 정상회담 준비 등을 위해 회담 개최 예정지인 싱가포르를 방문할 가능성도 제기되고 있다.\n\n최 부상은 지난 3월에도 아세안(ASEAN·동남아시아국가연합) 의장국이기도 한 싱가포르를 방문해 양국관계와 올해 8월 열리는 아세안지역안보포럼(ARF) 의제 등을 논의한 바 있다.\n\n또 지난해 북핵 문제를 두고 북미 간 긴장관계가 형성됐을 때도 ARF에 참석해 아세안을 상대로 여론전을 펼쳤다. 북한의 초청으로 비자이 쿠마르 싱 인도 외교부 국무장관이 방북했을 때도 최 부상은 싱 국무장관을 직접 영접하고, 한반도 문제를 논의하기도 했다.\n\n베이징 소식통은 "최 부상이 대(對)미 외교담당이 아니기 때문에 싱가포르로 갈 가능성이 큰 것은 아니다"며 "만약 싱가포르에 간다면 정상회담과 관련한 지원 작업 준비 등을 위한 것일 가능성이 크다"고 말했다.',
 '0']
```

train set의 첫번째 데이터를 보면 위와 같다.

&nbsp;





```python
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
```

가져온 vocab을 이용하여 토크나이저를 선언한다.

&nbsp;



```python
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
```

학습할 때 사용할 데이터셋 클래스를 선언한다.

&nbsp;



```python
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
```

파라미터를 세팅한다. 최대길이는 64로 설정했다. 

&nbsp;



```python
data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)
```

Dataset을 선언하고, Dataloader를 생성한다.

&nbsp;



## Make Model

```python
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=8,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out) 
```

분류에 사용할 모델을 만들어준다. 카테고리가 8개 이므로 num_classes는 8을 default로 설정한다.

&nbsp;



```python
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
```

모델, optimizer, loss function 등 학습에 필요한 것들을 선언한다.

&nbsp;



```python
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```

훈련중 정확도를 계산하기 위해 정확도를 계산하는 함수를 선언한다.

&nbsp;



## Train

```python
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, label)
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
```

```
epoch 1 train acc 0.1590909090909091
epoch 1 test acc 0.2760416666666667
epoch 2 train acc 0.4865056818181818
epoch 2 test acc 0.73828125
epoch 3 train acc 0.8160511363636364
epoch 3 test acc 0.85546875
epoch 4 train acc 0.9069602272727273
epoch 4 test acc 0.8645833333333334
epoch 5 train acc 0.9502840909090909
epoch 5 test acc 0.8854166666666666
epoch 6 train acc 0.9701704545454546
epoch 6 test acc 0.8697916666666666
epoch 7 train acc 0.9772727272727273
epoch 7 test acc 0.87890625
epoch 8 train acc 0.9879261363636364
epoch 8 test acc 0.8684895833333334
epoch 9 train acc 0.9928977272727273
epoch 9 test acc 0.8671875
epoch 10 train acc 0.9928977272727273
epoch 10 test acc 0.8684895833333334
```

모델을 훈련시킨다.

test set에 대한 정확도는 약 86.8%가 나왔다.  

&nbsp;



## Test

```python
def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100

def testModel(model, seq):
    cate = ["정치","경제","사회", "생활/문화","세계","기술/IT", "연예", "스포츠"]
    tmp = [seq]
    transform = nlp.data.BERTSentenceTransform(tok, max_len, pad=True, pair=False)
    tokenized = transform(tmp)

    modelload.eval()
    result = model(torch.tensor([tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device))
    idx = result.argmax().cpu().item()
    print("뉴스의 카테고리는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result,idx)))
```

직접 문장을 입력하고 결과를 확인하기 위한 함수를 선언한다.



```python
testModel(model, "신형 아이패드 프로에 m1칩 탑재 예정")
```

```
뉴스의 카테고리는: 기술/IT
신뢰도는: 97.48%
```

직접 입력한 문장에 대해 좋은 성능을 보이는 것을 확인 했다.

&nbsp;





## Conclusion

pretrain된 KoBERT모델을 이용하여 뉴스 데이터 카테고리 분류 task를 수행해본 결과 복잡하지 않은 과정으로도 좋은 성능을 확인할 수 있었다. 

별도의 tokenize 나 임베딩 과정을 따로 구현할 필요가 없음에도 좋은 성능을 보여서 가장 쉽게 테스트 해볼 수 있는 자연어 처리 방법인 것 같다.



