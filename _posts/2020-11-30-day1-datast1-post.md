---
title: "[인공지능 데브코스] 1주차 day1 - 자료구조(1)"
toc: true
toc_sticky: true
date: 2020-11-30
categories: 자료구조
---
## 11월 30일 월  

인공지능을 공부하기 전에 프로그래밍의 기본인 자료구조와 알고리즘의 기초부터 공부를 시작했다. 
대부분 익숙한 내용들이지만 오랬동안 다루지 않았던 Python으로 여러 자료구조들을 공부해 보면서 
Python에 대해 익숙해질 수 있었고 예전에 배웠던 것들을 확실하게 복습할 수 있었다.

## 자료구조
- 파이썬에서 이미 제공하는 데이터 타입: 문자열, 리스트, 딕셔너리, 튜플, 셋
- 자료구조를 알아야 하는 이유: 파이썬에서 제공하는 기본 데이터 타입으로 해결하기 힘든 문제를 해결하기 위해서  
=>풀어야 하는 문제에 따라 내가 사용하려는 자료구조가 어떤 성질을 가져야 하는지를 생각해야 한다.

해결하고자 하는 문제에 따라 최적의 해법은 서로 다르다  
=>이 선택을 어떻게 해야 하느냐를 알기 위해 자료구조를 이해해야 한다.

## 선형배열(Linear Arrays)
**배열**  
원소들을 순서대로 늘어 놓은것  

**리스트(list)**  
파이썬에서 제공하는 배열 데이터 타입  
(보통의 언어들과는 다르게 각 원소를 다른 데이터 타입으로 사용가능하다)  

**리스트의 연산**
- 원소 덧붙이기  

```python
list.append(val) #O(1)
```
- 끝에서 꺼내기  

```python
list.pop() #O(1)
```
- 원소 삽입하기  

```python
list.insert(index, val) #O(n)
```
- 원소 삭제하기

```python
del(list[index]) #항목으로 삭제
list.pop(index) #index로 삭제
#O(n)
```
- 원소 탐색하기

```python
list.index(val) #O(n)
```
<p>&nbsp;</p>  

**리스트의 정렬**  
- `sorted`: 내장함수, 정렬된 새로운 리스트를 얻어냄  
```python
nlist = sorted(list, reverse = True)
```
- `sort`: 리스트의 메서드, 해당리스트를 정렬  
```python
list.sort(reverse = True)
```
- `key`: lambda를 이용하여 원하는 기준으로 정렬  

```python
sorted(list, key = lambda x:len(x)) #길이를 기준으로 정렬
```
<p>&nbsp;</p>  

**리스트의 탐색**
- 선형탐색: 앞에서 부터 뒤로 발견될 때까지 순차적으로 탐색 `O(n)`  
- 이진탐색: 정렬되어있는 배열에서만 사용가능한 탐색 `O(log n)`  


**이진탐색의 구현**  
주어진 배열을 반으로 나누고 찾는 값이 왼쪽, 오른쪽중 어디에 있는지 확인 후 배열의 범위를 update해준다

```python
def solution(L, x):                   #L은 정렬된 배열, x는 찾고자하는 값
  left = 0; right = len(L)            #left, right를 각각 배열의 처음과 마지막으로 설정
  while left<right:                   #left가 right보다 작을 때까지 반복(범위안에 원소가 존재할 때까지)
      m = (left + right) // 2         #중간위치를 m으로 설정
      if L[m] < x:                    #중간위치보다 x가 크면 범위를 [m+1, right)로 update
          left = m+1
      elif L[m]:                      #중간위치보다 x가 작으면 범위를 [left, m)로 update
          right = m
  return left if (left<len(L) and L[left] == x ) else -1
  #범위안에 x가 있다면 left는 right와 같아져서 x를 가리키게 되고 아니라면 범위를 벗어나거나 다른 값을 가리키게 된다.
  #범위안에 x가 여러개라면 left는 그 중 가장 작은 값을 가리키고 범위안에 x가 없다면 x보다 큰 가장 작은 값을 가리킨다.
```
<p>&nbsp;</p>  


**이진탐색의 재귀적 구현**  
    
```python
def solution(L, x, l, u):         #L: 정렬된 배열, x: 찾는 값, l: update된 범위의 가장 처음위치, u: update된 범위의 가장 끝위치
  if l > u:                       #l 이 u보다 크거나 범위가 안에 원소가 없으므로 x를 찾지 못하고 재귀함수를 종료
      return -1
  mid = (l + u) // 2              #mid는 l과 u의 중간의 위치
  if x == L[mid]:                 #x가 mid가 가리키는 값과 같다면 찾은 것이므로 mid를 return
      return mid
  elif x < L[mid]:                #x가 중간위치보다 작다면 범위를 [l, mid)로 update
      return solution(L,x,l,mid)
  else:                           #아니라면 범위를 [mid+1, u)로 update
      return solution(L,x,mid+1,u)
```
<p>&nbsp;</p>  

## 연결리스트(Linked List)
**추상적 자료구조**  
내부구현은 숨겨두고 밖에서 직관적으로 사용할 수 있는 요소를 제공하는 자료구조  
내부구현을 잘 알지 못하더라도 사용하는 데에 지장이 없다.  
두가지 요소를 포함한다.  
  - Data: 정수, 문자열 등 자료가 담고있는 정보  
  - A set of operations: 삽입, 삭제, 순회, 정렬, 탐색 등 data를 다룰 수 있는 연산  

### 연결리스트  
값을 가지고 있는 노드들을 일렬로 연결시켜 만든 추상적 자료구조  

**연결리스트의와 배열의 차이**  
- 장점  
배열보다 중간에서의 원소의 삽입과 삭제가 빠르다

- 단점  
배열보다 탐색의 속도가 느리다.  
배열보다 공간을 더 많이 사용한다.

=> 원소의 삽입과 삭제가 많이 일어난다면 **연결리스트** 탐색이 많이 일어난다면 **배열**을 사용하는 것이 유리하다.

**Node**  
연결 리스트를 이루고 있는 하나의 단위를 노드라고 하며 노드 한개당 2가지 정보를 포함한다.  
- `data`: 노드 자체가 가지고 있는 값  
- `next`: 어떤 노드와 연결되어 있는지에 대한 정보  

```python
class Node:
  def __init__(self, item):
      self.data = item
      self.next = None
```
<p>&nbsp;</p>  

**연결 리스트의 구현**  
추상적 자료구조이므로 data, a set of operations를 포함하고 있다.  

**Data**  
- `head`: 맨 앞에 있는 노드  
- `tail`: 맨 뒤에 있는 노드  
- `nodeCount`: 연결 리스트의 길이  

```python
class LinkedList:
def __init__(self):
    self.nodeCount = 0
    self.head = None
    self.tail = None
```
<p>&nbsp;</p>  

**A set of operations**  
- `traverse()`모든 node의 값을 head부터 순차적으로 리스트에 담아서 반환 `O(n)`  

```python
def traverse(self):               
  result = []                     #빈 list 생성
  curr = self.head                #curr을 head로 초기화
  while curr is not None:         #curr이 None이 아닐 때까지 즉, 맨 끝에 도달할 때까지 반복
      result.append(curr.data)    #result에 curr의 값을 추가해주고
      curr = curr.next            #curr을 다음 노드로 변경
  return result
```
<p>&nbsp;</p>  

- `getAt(pos)`: pos위치에 있는 노드의 값을 반환 `O(n)`  

```python
def getAt(self, pos):               #O(n)
if pos < 1 or pos > self.nodeCount: #pos가 범위 밖에 있으면 None을 반환
    return None
i = 1
curr = self.head                    #curr을 맨 첫번째 노드로 설정
while i < pos:                      #curr을 pos번 next해줌
    curr = curr.next                
    i += 1
return curr                         #pos번째 node를 반환
```
<p>&nbsp;</p>  

- `insertAt(pos, newNode)`: pos위치에 newNode를 삽입 `O(n)`  

```python
def insertAt(self, pos, newNode):      
  if pos < 1 or pos > self.nodeCount + 1:  #pos가 범위 밖이라면 False를 반환 
      return False

  if pos == 1:                             #pos가 1이라면 맨 처음에 삽입하는 것이므로 head가 바뀌게 되어 따로 처리
      newNode.next = self.head             #newNode의 next를 지금 맨 처음 노드로 바꿔주고
      self.head = newNode                  #현재 head를 newNode로 바꿔줌

  else:                                    
      if pos == self.nodeCount + 1:        #link를 바꿔주어야 하기 때문에 삽입할 위치 전의 노드를 탐색
          prev = self.tail                 #속도향상을 위해 삽입할 위치가 맨 끝이라면 getAt을 사용하지 않고 tail을 사용
      else:                               
          prev = self.getAt(pos - 1)
      newNode.next = prev.next             #찾은 노드의 다음 노드를 새 노드의 next로 설정하고
      prev.next = newNode                  #찾은 노드의 다음을 새 노드로 설정

  if pos == self.nodeCount + 1:            #맨 끝에 삽입할 경우에는 tail이 바뀌므로 따로 처리
      self.tail = newNode

  self.nodeCount += 1                      #전체 노드의 개수를 1만큼 증가
  return True
```
 <p>&nbsp;</p>  

- `popAt(pos)`: pos위치에 있는 node를 삭제 `O(n)`  

```python
def popAt(self, pos):
  if pos < 1 and pos > self.nodeCount:      #범위밖이라면 error
      raise IndexError
  if pos == 1:                              #삭제할 위치가 1이라면 head가 바뀌므로 따로 처리
      curr = self.head
      data = curr.data
      self.head = curr.next
  else:
      prev = self.getAt(pos - 1)
      curr = prev.next
      data = curr.data
      prev.next = curr.next
      if pos == self.nodeCount:             #삭제할 위치가 마지막이라면 tail이 바뀌므로 따로처리
          self.tail = prev

  if pos == self.nodeCount and pos == 1:    #노드가 1개만 있는 경우에는 모든 노드가 사라지고 tail과 head도 None이 되므로 따로 처리
      self.tail = None
      self.head = None

  self.nodeCount -= 1                       #nodeCount를 1만큼 감소
  return data
```
<p>&nbsp;</p>  

**dummy node의 필요성**  
위에서 언급한 `popAt(pos)`, `insertAt(pos, newNode)`는 지정 위치까지 직접 이동해서 찾은 후 연산을 하기 때문에 선형 시간복잡도`O(n)`를 가지며 비효율적이다.  
연결리스트에서는 효율적인 삽입과 삭제를 위해 지정한 노드의 다음에 삽입하는 `insertAfter(prev, newNode)`와 지정한 노드 다음 노드를 삭제하는 `popAfter(prev)`를 사용하는데 이와 같은 함수로는 맨 앞에 노드 삽입이나 맨 앞 노드의 삭제를 할 수 없으므로 데이터를 가지고 있지 않은 dummy node를 맨앞에 추가하여 head로 사용하여 맨 앞 노드도 삭제, 삽입을 할 수 있도록 한다.
     
  - `insertAfter(prev, newNode)`: 지정한 노드 다음 위치에 새로운 노드를 삽입 `O(1)`  

```python
def insertAfter(self, prev, newNode):             #prev노드 다음위치에 newNode를 삽입하는 함수
    newNode.next = prev.next                      #새로운 노드의 다음을 prev의 다음이었던 노드로 설정
    if prev.next is None:                         #prev가 마지막 노드라면 tail이 새로운 노드가 되므로 따로 처리
        self.tail = newNode
    prev.next = newNode                           #prev의 next를 newNode로 설정
    self.nodeCount += 1
    return True
    
#insertAfter를 이용해서 insertAt을 다시 구현
def insertAt(self, pos, newNode):                 #insertAfter를 구현했으므로 pos를 이용해 삽입할 위치 전의 노드인 prev만 구하면 쉽게 구현할 수 있다.
    if pos < 1 or pos > self.nodeCount + 1:
        return False

    if pos != 1 and pos == self.nodeCount + 1:    #맨끝이라면 prev는 tail (굳이 필요는 없지만 시간을 단축하기 위해 추가)
        prev = self.tail
    else:                                         #아니라면  getAt을 이용해 구함
        prev = self.getAt(pos - 1)          
    return self.insertAfter(prev, newNode)        #위에서 구한 prev로 insertAfter를 이용해 노드 삽입
```
<p>&nbsp;</p>  

   - `popAfter(prev)`: 지정한 노드 다음위치의 노드를 삭제 `O(1)`  

  ```python
  def popAfter(self, prev):                       #prev다음위치에 있는 노드를 삭제하는 함수
    cur = prev.next                               #prev다음 위치에 있는 노드를 cur로 설정
    if cur is None:                               #prev가 맨 끝 노드였다면 삭제할 노드가 없으므로 따로 처리
        return None
    elif cur.next is None:                        #prev가 맨 끝에서 두번째 노드였다면 
        prev.next = None                          #맨 끝 노드를 삭제하는 것이므로 tail이 바뀌게 되어 따로 처리
        self.tail = prev
    else:                                         #그 이외의 경우는 prev.next를 다음다음 노드와 연결시키는 것으로 처리
        prev.next = cur.next

    self.nodeCount -= 1
    return cur.data
  
  #popAfter를 이용해서 popAt을 다시 구현
  def popAt(self, pos):                           #insert와 동일
    if pos < 1 or pos > self.nodeCount:
        raise IndexError
    else:
        prev = self.getAt(pos-1)
        return self.popAfter(prev)
  ```
<p>&nbsp;</p>  

위에서 구현한 `insertAt(pos, newNode)`와 `popAt(pos)`를 보면 처음에 구현했을때 보다 예외처리를 해야할 부분이 줄어서 코드가 간결해 진 것을 확인 할 수 있다.  
      
      
### 양방향 연결 리스트(Doubled Linked List)  
양쪽방향으로 진행가능한 Node를 이용하여 만든 연결 리스트로 앞, 뒤 방향으로 이동할 수 있다.  
**Node**  
```python
class Node:
  def __init__(self, item):
      self.data = item
      self.prev = None
      self.next = None
```
<p>&nbsp;</p>  

**Data**  
연결 리스트와 동일  

**A set of operations**  
- `getAt(pos)`: 단방향 연결 리스트와 다른점은 양방향으로 움직일 수 있기 때문에 pos가 head에 가까운지 확인후 더 가까운 쪽에서 탐색을 시작할 수 있다.  
  중간에 있는 값만 호출할 때는 소모시간이 같지만 평균 소모시간을 단축 시킬 수 있다. `O(n)`  

```python
def getAt(self, pos):
  if pos < 0 or pos > self.nodeCount:
      return None

  if pos > self.nodeCount // 2:
      i = 0
      curr = self.tail
      while i < self.nodeCount - pos + 1:
          curr = curr.prev
          i += 1
  else:
      i = 0
      curr = self.head
      while i < pos:
          curr = curr.next
          i += 1

  return curr
```
<p>&nbsp;</p>  

**양방향 연결 리스트에서의 dummy node**  
양방향 연결 리스트에서는 전에 구현했던 `insertAfter(pos, newNode)`, `popAfter(pos)` 뿐만 아니라 선택한 노드의 전 노드에서 작업을 수행하는 `insertBefore(pos, newNode)`, `popBefore(pos)` 또한 구현할 수 있다. 그러나 이를 위해서는 단방향 연결 리스트때와 같은 이유로 tail쪽에 값이 없는 dummy node가 필요하다. 아래에서는 head와 tail에 dummy node를 추가하고 단방향 연결 리스트에서 구현했던 연산들을 구현해 보았다.  

- 양방향 연결 리스트에서만 구현할 수 있는 `insertBefore(next, newNode)`와 다시 구현해본 `insertAt(pos, newNode)`  ㄱㄱㄱㄱㄱㄲㄱㄲㄱㄲㄲㄲㄲㄲ

```python
def insertBefore(self, next, newNode):           #link 바꾼 노드의 next와 prev를 모두 바꿔줘야 하므로 4개의 값을 바꿔야 한다.
  pre = next.prev
  pre.next = newNode
  newNode.next = next
  newNode.prev = pre
  next.prev = newNode
  self.nodeCount += 1
  return True

def insertAt(self, pos, newNode):
  if pos < 1 or pos > self.nodeCount + 1:
      return False

  next = self.getAt(pos + 1)
  return self.insertBefore(next, newNode)
```
<p>&nbsp;</p>  

- `popBefore(next)`와 다시 구현해본 `popAt(pos)`  

```python
def popBefore(self, next):
  cur = next.prev
  cur.prev.next = next
  next.prev = cur.prev
  self.nodeCount -= 1
  return cur.data


def popAt(self, pos):
    if pos < 1 or pos > self.nodeCount:
        raise IndexError
    else:
        next = self.getAt(pos+1)
        return self.popBefore(next)
```
<p>&nbsp;</p>  

위의 코드를 보면 양방향 연결 리스트를 이용하면 앞에 있는 노드를 삭제, 삽입 할 수 있을 뿐만 아니라 뒤에 있는 노드 또한 삭제, 삽입 할 수 있다는 것을 알 수 있다.  
정리해 보면 양방향 연결리스트는  

```
1. getAt(pos)함수의 소모시간을 단축시킬수 있다
2. 노드의 삭제 삽입을 좀 더 유연하게 할 수 있다
```
와 같은 장점들을 가지고 있는 것을 알 수 있다.  
이와 별개로 `insertAt(pos, newNode)`와 `popAt(pos)`를 보면 눈에 띄게 코드가 간결해 진 것을 확인 할 수 있는데 이는 dummy node의 추가에 의한 효과로 head와 tail이 바뀌지 않게 되면서 예외처리에 필요했던 if문들이 모두 사라진 것을 알 수 있다.  
이를 통해 특정 연산을 구현하는 역할 뿐만 아니라 실수의 최소화와 코드의 간결함을 위해 dummy node의 추가는 좋은 효과를 가지고 있는 것을 확인 할 수 있다.  

  

  

  

  

  

  

  

  

  

  

