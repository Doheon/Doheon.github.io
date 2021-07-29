---
title: "[코드구현] ChatBot - Transformer"
toc: true
toc_sticky: true
date: 2021-07-29
categories: Code-Implementation NLP
---

오픈된 데이터와 pytorch의 nn.Transformer를 이용하여 간단한 챗봇을 제작해보았다.  

아래의 강의를 참고하였으며 여기서 Tensorflow로 구현된 것을 참고하여 Pytorch로 다시 구현했다.

Code: <https://github.com/Doheon/Chatbot-Transformer>



Reference: <https://wikidocs.net/89786>

Data: <https://github.com/songys/Chatbot_data>



## Data Load

```python
import pandas as pd
import numpy as np
import re
```

```python
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()
```

|      | Q                         | A                   | label |
| ---- | ------------------------- | ------------------- | ----- |
| 0    | 12시 땡!                  | 하루가 또 가네요.   | 0     |
| 1    | 1지망 학교 떨어졌어       | 위로해 드립니다.    | 0     |
| 2    | 3박4일 놀러가고 싶다      | 여행은 언제나 좋죠. | 0     |
| 3    | 3박4일 정도 놀러가고 싶다 | 여행은 언제나 좋죠. | 0     |
| 4    | PPL 심하네                | 눈살이 찌푸려지죠.  | 0     |



데이터를 살펴보면 Q column에 질문이 있고, A column에 대답이 있는 형태이다. label은 어떤 종류의 질답인지에 대한 정보인데 여기서는 사용하지 않았다.



```python
questions = []
for sentence in train_data['Q']:
	# 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)
```

데이터를 list에 저장한다. !?. 와 같은 기호들은 공백을 추가해 준다.

&nbsp;



## Sentence Piece 학습 및 인코딩

서브 토큰 단위로 학습을 진행하기 위해 sentence piece를 학습시킨다.



```python
with open('all.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(questions))
    f.write('\n'.join(answers))
```

먼저 모든 문장을 하나의 txt 파일로 저장한다.

&nbsp;



```python
corpus = "all.txt"
prefix = "chatbot"
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

저장된 텍스트 파일을 이용하여 8000개의 vocab size를 가지고 사용자 지정 토큰 7개를 추가로 가지고 있는sentence piece를 학습시킨다.

&nbsp;



```python
vocab_file = "chatbot.model"
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
['▁안녕하세요', '▁만나서', '▁반갑', '습니다']
[4626, 1930, 4849, 154]
```

그 결과 학습이 잘 진행된 것을 확인 할 수 있다.

&nbsp;



```python
# 최대 길이를 40으로 정의
MAX_LENGTH = 40

START_TOKEN = [2]
END_TOKEN = [3]

# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(inputs, outputs):
    # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
    zeros1 = np.zeros(MAX_LENGTH, dtype=int)
    zeros2 = np.zeros(MAX_LENGTH, dtype=int)
    sentence1 = START_TOKEN + vocab.encode_as_ids(sentence1) + END_TOKEN
    zeros1[:len(sentence1)] = sentence1[:MAX_LENGTH]

    sentence2 = START_TOKEN + vocab.encode_as_ids(sentence2) + END_TOKEN
    zeros2[:len(sentence2)] = sentence2[:MAX_LENGTH]

    tokenized_inputs.append(zeros1)
    tokenized_outputs.append(zeros2)
  return tokenized_inputs, tokenized_outputs
```

학습된 sentence piece를 이용하여 주어진 문장을 정수로 인코딩하는 함수를 선언한다. 문장의 처음과 끝에는 sentence piece를 학습 시킬 때 따로 선언했던 START_TOKEN과 END_TOKEN의 index를 붙여준다.

&nbsp;



```python
questions_encode, answers_encode = tokenize_and_filter(questions, answers)
print(questions_encode[0])
print(answers_encode[0])
```

```
[   2 5566 6968 3210  111    3    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0]
[   2 5192  217 5936    7    3    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0]
```

그 결과 잘 인코딩 되어 저장된 것을 확인할 수 있다.

&nbsp;



## Dataset, DataLoader 생성

Batch 학습을 위해 dataset, dataloader를 생성한다.



```python
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, questions, answers):
        questions = np.array(questions)
        answers = np.array(answers)
        self.inputs = questions
        self.dec_inputs = answers[:,:-1]
        self.outputs = answers[:,1:]
        self.length = len(questions)
    
    def __getitem__(self,idx):
        return (self.inputs[idx], self.dec_inputs[idx], self.outputs[idx])

    def __len__(self):
        return self.length

BATCH_SIZE = 64
dataset = SequenceDataset(questions_encode, answers_encode)
dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
```

dataset은 3개의 값으로 이루어지도록 선언한다. 첫번째 값은 주어진 질문이고, 두번째 값은 디코더의 입력으로 마지막 토큰값이 제거된 대답이고, 마지막 값은 첫 토큰값이 제거된 결과이다.

생성된 dataset을 이용하여 64 의batch size를 가지고 있는 dataloader를 생성한다.

&nbsp;



## Model 생성

훈련을 진행할 Model을 선언했다. nn.Transformer를 사용하여 만들었다.



```python
from torch.nn import Transformer
from torch import nn
import torch
import math

class TFModel(nn.Module):
    #ntoken: vocab의 size
    #ninp: embedding할 차원의 크기
    #nhead: num head
    #nhid: feedforward의 차원
    #nlayers: layer의 개수
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.transformer = Transformer(ninp, nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, num_decoder_layers=nlayers,dropout=dropout)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.pos_encoder_d = PositionalEncoding(ninp, dropout)
        self.encoder_d = nn.Embedding(ntoken, ninp)

        self.ninp = ninp
        self.ntoken = ntoken

        self.linear = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, srcmask, tgtmask, srcpadmask, tgtpadmask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder_d(tgt)


        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), srcmask, tgtmask, src_key_padding_mask=srcpadmask, tgt_key_padding_mask=tgtpadmask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask
```



nn.Transformer를 사용했고, nn.Transformer에는 positional encoding과 Embedding이 포함되어 있지 않기 때문에 인코더와 디코더에 각각 직접 선언하여 포함시켜 주었다.

입력한 텐서와 같은 크기의 attention mask를 생성해주는 gen_attention_mask 함수도 선언했다.

&nbsp;



## Train

선언한 모델 클래스를 이용하여 모델을 생성하고 훈련을 진행했다.



```python
device = torch.device("cuda")

lr = 1e-4
model = TFModel(vocab_size+7, 256, 8, 512, 2, 0.2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

&nbsp;



```python
epoch = 50
from tqdm import tqdm

model.train()
for i in range(epoch):
    batchloss = 0.0
    progress = tqdm(dataloader)
    for (inputs, dec_inputs, outputs) in progress:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(MAX_LENGTH).to(device)
        src_padding_mask = gen_attention_mask(inputs).to(device)
        tgt_mask = model.generate_square_subsequent_mask(MAX_LENGTH-1).to(device)
        tgt_padding_mask = gen_attention_mask(dec_inputs).to(device)

        result = model(inputs.to(device), dec_inputs.to(device), src_mask, tgt_mask, src_padding_mask,tgt_padding_mask)
        loss = criterion(result.permute(1,2,0), outputs.to(device).long())
        progress.set_description("{:0.3f}".format(loss))
        loss.backward()
        optimizer.step()
        batchloss += loss
    print("epoch:",i+1,"|","loss:",batchloss.cpu().item() / len(dataloader))
```

```
epoch: 20 | loss: 0.5513939110008446
0.507: 100%|██████████| 185/185 [00:08<00:00, 21.30it/s]
epoch: 21 | loss: 0.5286682541305955
0.480: 100%|██████████| 185/185 [00:08<00:00, 21.46it/s]
epoch: 22 | loss: 0.5066682970201647
0.460: 100%|██████████| 185/185 [00:08<00:00, 21.40it/s]
epoch: 23 | loss: 0.48444976806640627
0.428: 100%|██████████| 185/185 [00:08<00:00, 22.04it/s]
epoch: 24 | loss: 0.4617009755727407
0.467: 100%|██████████| 185/185 [00:08<00:00, 22.28it/s]
epoch: 25 | loss: 0.43948953989389783
0.500: 100%|██████████| 185/185 [00:08<00:00, 21.45it/s]
epoch: 26 | loss: 0.4179861945074958
0.331: 100%|██████████| 185/185 [00:08<00:00, 22.72it/s]
epoch: 27 | loss: 0.39681738776129644
0.342: 100%|██████████| 185/185 [00:08<00:00, 20.57it/s]
epoch: 28 | loss: 0.37577456912478885
0.333: 100%|██████████| 185/185 [00:08<00:00, 21.42it/s]
epoch: 29 | loss: 0.35584098197318415
0.345: 100%|██████████| 185/185 [00:08<00:00, 21.59it/s]
epoch: 30 | loss: 0.33466512319203967
```

30번의 epoch로 학습을 진행했다.

&nbsp;



## Evaluate

실제 문장을 넣었을 때 적절한 대답이 나오는지 확인해 보았다.



```python
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    input = torch.tensor([START_TOKEN + vocab.encode_as_ids(sentence) + END_TOKEN]).to(device)
    output = torch.tensor([START_TOKEN]).to(device)

    # 디코더의 예측 시작
    model.eval()
    for i in range(MAX_LENGTH):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        src_padding_mask = gen_attention_mask(input).to(device)
        tgt_padding_mask = gen_attention_mask(output).to(device)

        predictions = model(input, output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).transpose(0,1)
        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = torch.LongTensor(torch.argmax(predictions.cpu(), axis=-1))


        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if torch.equal(predicted_id[0][0], torch.tensor(END_TOKEN[0])):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = torch.cat([output, predicted_id.to(device)], axis=1)

    return torch.squeeze(output, axis=0).cpu().numpy()

def predict(sentence):
    prediction = evaluate(sentence)
    predicted_sentence = vocab.Decode(list(map(int,[i for i in prediction if i < vocab_size+7])))

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence
```

&nbsp;



```python
result = predict("놀고싶다")
```

```
Input: 놀고싶다
Output: 저도요 ! ! !
```

&nbsp;



```python
result = predict("배고파")
```

```
Input: 배고파
Output: 얼른 뭐라도 드세요 .
```

&nbsp;



```python
result = predict("고민 상담 해줘")
```

```
Input: 고민 상담 해줘
Output: 네 말씀하세요 .
```

&nbsp;



```python
result = predict("난 뭘 해야 할까?")
```

```
Input: 난 뭘 해야 할까?
Output: 가장 중요한 것 같아요 .
```

&nbsp;



직접 실험해 본 결과 나쁘지는 않지만 그렇게 좋은 성능을 가지고 있진 않았다.

결과가 train epoch에 민감하며 hyper parameter tuning이 추가로 필요해 보인다.



