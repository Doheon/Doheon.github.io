---
title: "[코드구현] Sentence Classification - FastText+LSTM"
toc: true
toc_sticky: true
date: 2021-07-27
categories: Code-Implementation NLP
---

뉴스의 카테고리를 분류하는 작업을 토큰화, 단어 임베딩을 거친 후에 LSTM과 FC layer를 사용하는 방법으로 수행해 보았다.

토큰화와 임베딩을 다양한 방법으로 해보면서 최적의 방법을 찾아보았다.

Code: <https://github.com/Doheon/NewsClassification-LSTM>



사용한 데이터셋은 아래와 같다.

Dataset: <http://ling.snu.ac.kr/class/cl_under1801/FinalProject.htm>

&nbsp;



## Import Module

```python
from gensim.models import FastText
from konlpy.tag import Hannanum
import sentencepiece as spm

from tqdm import trange
import os
```

단어 임베딩은 FastText를 이용해 진행하였고, 토큰화는 sentence piece와 형태소 분석 두가지 방법으로 진행하였다.

&nbsp;



## Load Data

```python
dataset_train = []
dataset_test = []
dataset_all = []

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
        dataset_all.append(strings)
        file.close()

print(len(dataset_train), len(dataset_test))
```

```
1360 240
```



파일을 읽어서 기사 내용과 카테고리를 하나의 데이터로 list를 만든다.

각 카테고리당 170개를 train set, 30개를 test set으로 사용했다.

모든 데이터는 dataset_all이라는 list에 따로 저장한다.

&nbsp;



## Tokenize

### 형태소 분석

먼저 형태소 분석 모듈인 hannanum을 사용하여 문장들을 토큰화 한 후 그 결과로 단어 임베딩 모델인 FastText를 학습시켰다.



```python
hannanum = Hannanum()
vocab_morphs = set()
tokened_morphs = []
with trange(len(dataset_all)) as tr:
    for i in tr:
        morphs = hannanum.morphs(dataset_all[i])
        for morph in morphs:
            vocab_morphs.add(morph)
        tokened_morphs.append(morphs)
```

토큰화된 모든 문장들을 list에 저장한다.

&nbsp;



```python
emb_num = 128

embedding = FastText(tokened_morphs, vector_size=emb_num, window=12, min_count=5, sg=1)
embedding.save("fasttext_morph.model")
```

토큰화된 결과를 이용하여 FastText를 학습시킨다.

&nbsp;



```python
model_morphs = FastText.load("fasttext_morph.model")
model_morphs.wv.most_similar("국회의원")
```

```
[('의원직', 0.9015116095542908),
 ('국회의장', 0.8992209434509277),
 ('사직', 0.8932391405105591),
 ('출마', 0.8877411484718323),
 ('사직서', 0.8795743584632874),
 ('현역의원', 0.8716889023780823),
 ('사퇴', 0.8710533976554871),
 ('의원들', 0.8594748973846436),
 ('보궐선거', 0.8586255311965942),
 ('현역의원들', 0.8532490134239197)]
```

학습 결과 어느정도 훈련이 잘 진행된 것을 확인 할 수 있다.

&nbsp;



### Sentence Piece

토큰화 모듈인 Sentence Piece를 사용해서 문장들을 토큰화 한 후 그 결과를 이용하여 단어 임베딩 모델인  FastText를 학습시켰다.



```python
f = open("allsentence.txt","w")
f.write("".join(dataset_all).replace("\xa0", ""))
f.close()
```

일단 모든 데이터들을 하나의 텍스트 파일에 저장한다.

&nbsp;



```python
corpus = "allsentence.txt"
prefix = "news"
vocab_size = 8000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰
```

생성된 텍스트 파일을 이용하여 sentence piece를 학습시킨다.

&nbsp;



```python
vocab_file = "news.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)
line = "안녕하세요 만나서 반갑습니다"
pieces = vocab.encode_as_pieces(line)
ids = vocab.encode_as_ids(line)
print(line)
print(pieces)
print(ids)
```

```
안녕하세요 만나서 반갑습니다
['▁안', '녕', '하', '세요', '▁만나', '서', '▁반', '갑', '습니다']
[89, 7577, 6518, 2892, 957, 6521, 126, 7021, 107]
```

확인해본 결과 토큰화가 이루어 지는 것을 확인할 수 있다.

&nbsp;





```python
tokened_sp = []
with trange(len(dataset_all)) as tr:
    for i in tr:
        tokened_sp.append(vocab.encode_as_pieces(dataset_all[i]))
```

모든 문장들을 학습된 sentence piece를 이용하여 토큰화 하고 list에 저장한다.

&nbsp;



```python
emb_num = 128

embedding = FastText(tokened_sp, vector_size=emb_num, window=10, min_count=2, sg=1)
embedding.save("fasttext_sp.model")
```

저장된 list를 이용하여 FastText를 학습시킨다.

&nbsp;



```python
model_sp = FastText.load("fasttext_sp.model")
model_sp.wv.most_similar("국회의원")
```

```
[('▁국회의원', 0.9339150786399841),
 ('의원', 0.820768415927887),
 ('▁출마', 0.8091025948524475),
 ('▁현역', 0.7669360041618347),
 ('▁지방선거에', 0.762986421585083),
 ('궐선거', 0.7626964449882507),
 ('▁사직서', 0.7208353281021118),
 ('▁송파을', 0.7189249396324158),
 ('▁사직', 0.7137511372566223),
 ('▁의원', 0.713568389415741)]
```

결과를 확인해 보면 잘 학습이 된 것을 확인할 수 있다.

형태소 분석때와는 다르지만 어느정도 비슷한 단어가 올라온 것을 확인할 수 있다.

&nbsp;



## Dataset, DataLoader 생성

배치 학습을 위해 Dataset, DataLoader를 생성한다.



```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda")
```

필요한 모듈들을 import한다.

&nbsp;



```python
class SentenceDataset(Dataset):
    def __init__(self, dataset, tokenizer, fasttextModel, max_len):
        self.sentences = []
        with trange(len(dataset)) as tr:
            for i in tr:
                sen = dataset[i][0]
                sen = tokenizer(sen) #토큰화
                if len(sen) < max_len:
                    sen = sen + (max_len-len(sen)) * [""]
                sen = sen[:max_len]
                sen = fasttextModel[sen] #임베딩
                self.sentences.append(sen)
        self.labels = [np.int32(i[1]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i],self.labels[i])

    def __len__(self):
        return (len(self.labels))
```

tokenizer와 FastText Model을 넣어주면 그에 맞게 단어들을 임베딩해주는 Dataset을 선언한다.

&nbsp;



```python
max_len = 32
sp_train = SentenceDataset(dataset_train, vocab.encode_as_pieces, model_sp.wv, max_len)
sp_test = SentenceDataset(dataset_test, vocab.encode_as_pieces,model_sp.wv, max_len)
```

```python
hannanum = Hannanum()
morphs_train = SentenceDataset(dataset_train, hannanum.morphs, model_morphs.wv, max_len)
morphs_test = SentenceDataset(dataset_test, hannanum.morphs,model_morphs.wv, max_len)
```

sentence piece와 형태소 분석 두 가지의 방법으로 데이터 셋을 생성한다.

&nbsp;





```python
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(morphs_train, batch_size=batch_size, num_workers=5, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(morphs_test, batch_size=batch_size, num_workers=5, shuffle=True)
```

생성한 데이터 셋으로 DataLoader를 생성한다.

일단은 형태소 분석 Dataset을 사용하여 DataLoader를 생성한다.

&nbsp;



## Model 생성

훈련의 사용할 모델을 생성한다.



```python
class LSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, num_layers = 1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)

        self.linear = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
    
        self.fc = nn.Sequential(self.linear, self.dropout, self.relu, self.linear2, self.dropout)

    def forward(self, x_input):
        lstm_out, (h,c) = self.lstm(x_input)
        output = self.fc(lstm_out[:,-1,])
        return output
```

LSTM layer를 거친 후 LSTM layer의 마지막 output 값이 FC layer를 두개를 통과하는 간단한 모델을 선언한다.

최종적으로 num_classes만큼의 결과가 나오도록 FC layer를 생성해 준다.

&nbsp;





```python
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```

정확도를 측정할 수 있도록 하는 함수를 선언한다.

&nbsp;



## Training

```python
lstm = LSTM(emb_num, 8, 128, 2).to(device)

optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.0003)
criterion = nn.CrossEntropyLoss()
epochs = 100


with trange(epochs) as tr:
    for i in tr:
        itloss = 0
        trainacc = 0
        testacc = 0
        
        lstm.train()
        for batch_id, (input, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input = input.to(device)
            label = label.long().to(device)
            out = lstm(input)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            itloss += loss.cpu().item()
            trainacc += calc_accuracy(out,label)


        lstm.eval()
        for batch_idt, (input, label) in enumerate(test_dataloader):
            input = input.to(device)
            label = label.long().to(device)
            out = lstm(input)
            testacc += calc_accuracy(out,label)

        tr.set_postfix(trainacc="{0:.3f}".format(trainacc/(batch_id+1)), loss="{0:.3f}".format(itloss/(batch_id+1)),  testacc="{0:.3f}".format(testacc/(batch_idt+1)))
```

```
100%|██████████| 100/100 [01:30<00:00,  1.10it/s, loss=0.432, testacc=0.723, trainacc=0.822]
```

CrossEntropyLoss를 loss function으로 사용하여 훈련을 진행한다.

test accuarcy는 0.723이 나왔다.

&nbsp;



## Hyperparameter Tuning

최적의 결과를 얻어내기 위해 max_len, epochs, lstm layer개수, fc layer개수, token화 방법, embedding 방법과 같은 hyper parameter들을 변경해 보면서 학습을 진행해 보았다.

결과는 아래와 같다.

| no   | result                                    | max_len | epochs | lstm_n      | fc_n | token  | embedding    |
| ---- | ----------------------------------------- | ------- | ------ | ----------- | ---- | ------ | ------------ |
| 1    | loss=0.189, testacc=0.755, trainacc=0.907 | 64      | 200    | 2           | 2    | sp     | fasttext     |
| 2    | loss=0.292, testacc=0.788, trainacc=0.857 | 64      | 200    | 2           | 2    | morphs | fasttext     |
| 3    | loss=0.539, testacc=0.818, trainacc=0.782 | 64      | 100    | 2           | 2    | morphs | fasttext     |
| 4    | loss=0.401, testacc=0.772, trainacc=0.842 | 64      | 100    | 2           | 1    | morphs | fasttext     |
| 5    | loss=0.736, testacc=0.788, trainacc=0.749 | 64      | 100    | 1           | 2    | morphs | fasttext     |
| 6    | loss=0.512, testacc=0.796, trainacc=0.795 | 64      | 100    | 2(concat h) | 2    | morphs | fasttext     |
| 7    | loss=0.453, testacc=0.729, trainacc=0.813 | 32      | 100    | 2           | 2    | morphs | fasttext     |
| 8    | loss=0.145, testacc=0.501, trainacc=0.900 | 64      | 100    | 2           | 2    | sp     | nn.Embedding |





**best: no.3, loss=0.539, testacc=0.818, trainacc=0.782** 

토큰화 방법은 sentence piece보다 형태소 분석이 더 좋은 결과가 나왔다. 한국어에 대해서는 형태소 분석이 더 좋은 성능을 가지고 있다고 생각된다.

임베딩방법은 FastText와 torch의 nn.Embedding로 두가지 방법으로 진행해 보았다. nn.embedding을 사용하면 trainacc만 높게나오는 오버피팅이 되는 경향이 있어서 FastText를 사용한 임베딩 방법이 더 좋은 성능을 보였다.

&nbsp;



## Evaluate

```python
cate = ["정치","경제","사회", "생활/문화","세계","기술/IT", "연예", "스포츠"]
def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    
    tmp = []
    for i in valscpu:
        tmp.append(((np.exp(i))/a).item() * 100)
    print(["{}:{:.2f}%".format(cate[i],v) for i,v in enumerate(tmp)])

    return ((np.exp(valscpu[idx]))/a).item() * 100

def test_model(seq, model, tokenizer, fasttextmodel):
    sen = tokenizer(seq)
    # sen = vocab.encode_as_ids(seq)
    if len(sen) < max_len:
        # sen = sen + (max_len-len(sen)) * [1]
        sen = sen + (max_len-len(sen)) * [""]
    sen = sen[:max_len]
    sen = fasttextmodel[sen]
    sen = torch.tensor(sen).unsqueeze(0).to(device)
    model.eval()
    result = model(sen)
    idx = result.argmax().cpu().item()
    print("뉴스의 카테고리는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result,idx)))
```

직접 문장을 넣었을 때 결과를 바로 확인 할 수 있도록 테스트를 할 수 있는 함수를 선언한다.

&nbsp;



```python
test_model("신형 아이패드 프로에 m1칩 탑재 예정", lstm, hannanum.morphs, model_morphs.wv)
```

```
뉴스의 카테고리는: 기술/IT
['정치:0.00%', '경제:0.16%', '사회:0.15%', '생활/문화:3.64%', '세계:0.12%', '기술/IT:94.37%', '연예:0.13%', '스포츠:1.42%']
신뢰도는: 94.37%
```

직접 테스트해본 결과 어느정도는 좋은 성능을 보였지만 64길이로 학습을 시키고 짧은 길이로 테스트를 해서인지 test accuracy만큼 좋은 성능을 가지고 있진 않았다.

&nbsp;
