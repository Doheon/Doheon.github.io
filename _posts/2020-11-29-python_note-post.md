---
title: "파이썬 노트"
toc: true
toc_sticky: true
date: 2020-11-29
categories: 노트
---

#### 파이썬에서 적고싶은거 발견하면 아무거나 적는 곳  

아무거나 아무순서로 적었다.  

#### dict.get(a,b)
dircionary를 사용할 때 key가 없는 경우에도 사용가능한 함수  
key가 없을 때 return할 값을 b로 미리 설정해 놓는다.  

#### heapq  
최소힙만 지원한다. (최대힙을 사용하고 싶다면 마이너스 값을 저장하는 등의 추가과정 필요)  
따로 자료구조를 제공하는게 아니라 리스트를 그대로 이용한다.  
import heapq  
heapq.heapify(list) : 이미 있던 리스트를 heapq화  
heapq.heappop(list) : 최소원소를 꺼내고 리턴  
heapq.heappush(list, val) : 원소를 넣어줌  

from heapq import heapify, heappush, heappop  
이걸로 해주면 귀찮게 heapq를 일일히 붙여줄 필요가 없다.  


#### for else구문  
```python
for a in listx:  
  if a==b  
    break  
else:  
    a+=1  
```

if는 for문 안에 있지만 else는 for문 밖에 있는 특이한 구조  
if에 break가 있어야 하며 break를 만나면 else로 간다.  
c에서 for문의 index를 따로 저장해서 했었던걸 하게 해준다.   


#### deque
stack과 queue를 동시해 할 수 있는 자료구조  
bfs할 때 사용하자(큐가 필요할 때)  

from collections import deque  

함수들  
append(), pop(), appendleft(), popleft(), insert(), remove(), reverse()  


#### 파이썬에서의 재귀함수  
파이썬은 재귀스택이 1000번으로 제한되어 있다.  
dfs를 쓸때는 보통 이를 초과하는 경우가 생기므로  
```python
import sys
sys.sertrecursionlimit(30000)
```
를 맨 위에 적어놓자  


#### map  
데이터를 원하는 형태의 자료형으로 바꿔주는 함수  
list()를 또한번 해줘야 원하는 형태가 된다.  
numbers = list(map(str, numbers))

#### set
set은 mutable한 값을 원소로 가질 수 없다.  
set에 list를 넣고 싶을 때는 tuple로 바꿔주고 넣어야 한다.  

#### strip(), split()
문자열에서 사용하는 함수  
strip(): 양끝의 여백을 없애준다.  
split(','): 입력값을 기준으로 문자열을 리스트로 나눠준다.  


### format()
print("나는 {}등이다".format(rank))  
{}안에 들어갈 내용을 정해줄 수 있음  

### dir(class)  
모든 매서드 확인  

### os  
import os  
os.path.join(BASE_DIR, “homepage”, “template”)  
경로들을 합쳐준다. (슬래시를 자동으로 해준다.)  




