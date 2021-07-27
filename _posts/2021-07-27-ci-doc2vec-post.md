---
title: "[코드구현] Sentence Classification - Doc2Vec
toc: true
toc_sticky: true
date: 2021-07-27
categories: Code-Implementation NLP
---

sentencepiece와 gensim 모듈의 Doc2Vec를 사용해서 문장을 임베딩하고 그 결과에 pytorch를 사용한 classifier를 사용하여 뉴스 데이터의 카테고리를 분류하는 task를 수행해 보았다.



사용한 데이터셋은 아래와 같다.

Dataset: <http://ling.snu.ac.kr/class/cl_under1801/FinalProject.htm>





## Import Module

```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import sentencepiece as spm

from tqdm import trange
import os
```

먼저 gesim의 Doc2Vec를 import한다. 

단어는 단어 단위로 나누지 않고 sentence piece를 이용한 sub token 단위로 토큰화 해서 사용한다.







## Load Data

8개의 클래스를 가진 뉴스 기사 데이터를 읽어온다.



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
        file = open(fname, "r", encoding="utf8")
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

파일을 읽어서 기사 내용 문자열과 해당하는 카테고리를 하나의 데이터로 list를 만든다.

각 카테고리당 200개의 데이터 중 170개를 train set, 30개를 test set으로 사용한다.







## Train Sentence Piece

Doc2Vec를 학습시킬 때 subtoken 단위로 학습을 시키기 위해 Sentence Piece를 먼저 학습을 시킨다.



```python
f = open("allsentence.txt","w")
f.write("".join(dataset_all).replace("\xa0", ""))
f.close()
```

모든 문장을 하나의 파일로 합친 후 allsentence.txt 라는 파일로 저장한다.





```python
#subword 단위
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

생성된 텍스트 파일을 사용해서 sentence piece를 학습 시킨다.





```python
vocab_file = "news.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)
cab_file = "news.model"
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

학습 결과 subtoken 단위로 잘 토큰화 되는 것을 확인 했다.





```python
tokened_sp = []
with trange(len(dataset_all)) as tr:
    for i in tr:
        tokened_sp.append(vocab.encode_as_pieces(dataset_all[i]))
```

학습된 sentence piece를 사용하여 모든 문장을 토큰화 한하고 새로운 list에 저장한다.





## Train Doc2Vec

```python
class Doc2VecCorpus:
    def __iter__(self):
        for idx, doc in enumerate(tokened_sp):
            yield TaggedDocument(
                words = doc, 
                tags = [idx])

doc2vec_corpus = Doc2VecCorpus()
```

doc2vec를 학습시킬 corpus로 Doc2VecCorpus라는 class를 선언한다.





```python
embed_num = 128
doc2vec_model = Doc2Vec(documents = doc2vec_corpus,dm=2,  vector_size=embed_num, window = 10, min_count = 5)
```

생성한 corpus를 이용하여 128차원의 임베딩으로 Doc2vec model을 학습 시킨다.





## Train Classifier

pytorch를 사용해서 학습된 기사들의 임베딩을 통해 multiclass classifier를 학습 시킨다.



```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
device = torch.device("cuda")
```



```python
class SentenceDataset(Dataset):
    def __init__(self, dataset, tokenizer, doc2vecmodel, max_len):
        self.sentences = []
        with trange(len(dataset)) as tr:
            for i in tr:
                sen = dataset[i][0]
                sen = tokenizer(sen)
                l = min(max_len, len(sen))
                sen = sen[:l]
                sen = doc2vecmodel.infer_vector(sen)
                self.sentences.append(sen)
        self.labels = [np.int32(i[1]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i],self.labels[i])

    def __len__(self):
        return (len(self.labels))
```

torch의 Dataset을 사용하여 문장의 임베딩과 라벨을 가지고 있는 Dataset class를 선언한다.





```python
max_len = 512
data_train = SentenceDataset(dataset_train, vocab.encode_as_pieces,doc2vec_model, max_len)
data_test = SentenceDataset(dataset_test, vocab.encode_as_pieces,doc2vec_model, max_len)
```

선언한 SentenceDataset을 이용하여 train, test dataset을 생성한다.





```python
batch_size = 64
train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)
```

batch 학습을 위해 생성한 dataset을 가지고 Dataloader를 생성한다.





```python
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.linear = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
    
        self.fc = nn.Sequential(self.linear, self.dropout, self.relu, self.linear2, self.dropout)

    def forward(self, x_input):
        output = self.fc(x_input)
        return output
```

FC layer를 두개를 가지고 있는 multiclass classifier를 선언한다. 문장 임베딩을 받아서 바로 분류하는 간단한 모델을 사용했다.





```python
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```

훈련중 정확도를 계산할 함수를 선언한다. 가장 큰 값을 가지고 있는 index가 label과 일치하는지 확인한다.





```python
model = Classifier(embed_num, 8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
criterion = nn.CrossEntropyLoss()
epochs = 100

with trange(epochs) as tr:
    for i in tr:
        itloss = 0
        trainacc = 0
        testacc = 0
        
        model.train()
        for batch_id, (input, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input = input.to(device)
            label = label.long().to(device)
            out = model(input)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            itloss += loss.cpu().item()
            trainacc += calc_accuracy(out,label)


        model.eval()
        for batch_idt, (input, label) in enumerate(test_dataloader):
            input = input.to(device)
            label = label.long().to(device)
            out = model(input)
            testacc += calc_accuracy(out,label)

        tr.set_postfix(trainacc="{0:.3f}".format(trainacc/(batch_id+1)), loss="{0:.3f}".format(itloss/(batch_id+1)),  testacc="{0:.3f}".format(testacc/(batch_idt+1)))
```

```
100%|██████████| 100/100 [00:41<00:00,  2.38it/s, loss=0.463, testacc=0.810, trainacc=0.823]
```

optimizer는 Adam optimizer를 사용했고, multi class classification이기 때문에 loss function 으로는 CrossEntropyLoss를 사용했다.



최종 test accuracy는 0.81이 나왔다.

RNN없이 간단하게 임베딩과 fc layer만 사용한 것 치고는 생각보다 높은 정확도가 나왔다.

생각보다 Doc2Vec의 임베딩 능력이 좋은 것 같다고 느껴졌다.





## Test

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

def test_model(seq, model):
    model.eval()
    sen = vocab.encode_as_pieces(seq)
    l = min(max_len, len(sen))
    sen = sen[:l]
    sen = doc2vec_model.infer_vector(sen)
    sen = torch.tensor(sen).unsqueeze(0).to(device)
    result = model(sen)
    idx = result.argmax().cpu().item()
    print("뉴스의 카테고리는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result,idx)))
```

직접 타이핑한 문장의 카테고리를 바로 출력해주는 함수를 선언했다.



```python
test_model("신형 아이패드 프로에 m1칩 탑재 예정", model)
```

```
뉴스의 카테고리는: 기술/IT
['정치:1.18%', '경제:10.40%', '사회:14.44%', '생활/문화:16.44%', '세계:5.13%', '기술/IT:30.20%', '연예:6.33%', '스포츠:15.89%']
신뢰도는: 30.20%
```

정답은 어느 정도 맞췄지만 정확도가 그렇게 높지 않은 것을 확인했다.

아마 임베딩을 학습 할 때는 긴 문장으로 했지만 테스트의 문장은 길이가 짧아서 별로 좋지 않은 결과가 나온 것으로 예상된다.



## Result

**정확도: 0.81**

