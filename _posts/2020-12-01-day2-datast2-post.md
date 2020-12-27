---
title: "[인공지능 데브코스] 1주차 day2 - 자료구조(2)"
toc: true
toc_sticky: true
date: 2020-12-01
categories: TIL 자료구조
---
## 12월 01일 화

어제 공부했던 자료구조인 리스트, 연결 리스트에 이어서 자료구조를 공부했다. 
스택, 큐, 트리, 힙에 대해 공부했고 자료구조에 대한 구현과 자료구조를 이용해서 풀수 있는 예제들을 풀어보았다. 
여러 자료구조들을 복습해보면서 개념을 정리하는 데에 도움이 되었고 Python을 다루는 것이 조금 더 익숙해 진것 같다. 
그러나 예제를 풀면서 발생한 대부분의 에러들이 C와 Python의 문법을 혼동하여 생긴 것인만큼 아직도 헷갈리는 부분이 많은 것 같다.

## 스택(Stacks)  

자료를 보관할 수 있는 선형 구조
- 넣을 때는 한쪽 끝에서 밀어넣어야 하고 (push)
- 꺼낼 때는 같은 쪽에서 뽑아 꺼내야 한다 (pop)  
=> LIFO(last in last out)의 특징을 가지고 있는 선형 자료구조이다.

**연산의 정의**
- `size()`: 스택의 전체 크기를 반환
- `isEmpty()`: 스택이 비어있는지 여부를 반환
- `push(val)`: 스택의 맨 끝에 값을 추가
- `pop()`: 스택의 맨 끝 값을 삭제 후 그 값을 반환
- `peek()`: 스택의 맨 끝 값을 반환

### 스택의 구현
연결 리스트 각 요소인 노드를 직접 구현해서 만들었지만 스택은 배열 혹은 연결리스트를 이용해서 구현 할 수 있다.  

1. 배열을 이용하여 구현  
Python 리스트와 메서드들을 이용하여 구현할 수 있다.

```python
class ArrayStack:
    def __init__(self):
        self.data = []

    def size(self):                   #O(1)
        return len(self.data)
  
    def isEmpty(self):                #O(1)
        return self.size() == 0

    def push(self, item):             #O(1)
        self.data.append(item)

    def pop(self):                    #O(1)
        return self.data.pop()

    def peek(self):                   #O(1)
        return self.data[-1]
```
<p>&nbsp;</p>  

2. 연결 리스트를 이용하여 구현  
양방향 연결 리스트를 이용하여 구현 할 수 있다.  

```python
class dLinkedListStack:
    def __init__(self):                             #O(1)
        self.data = DoublyLinkedList()
    
    def size(self):                                 #O(1)
        return self.data.getLength()
    
    def isEmpty(self):                              #O(1)
        return self.size() == 0
    
    def push(self,item):                            #O(1)
        node = Node(item)
        self.data.insertAt(self.size() + 1, node)
    
    def pop(self):                                  #O(1)
        return self.data.popAt(self.size())
        
    def peek(self):                                 #O(1)
        return self.data.getAt(self.size()).data
```
위의 코드에서 양방향 연결 리스트의 `insertAt`,  `popAt`과 `getAt`은 시간 복잡도가 `O(n)`으로 알고 있지만 `O(1)`인 이유는 양방향 리스트는 알아서 head와 tail중 가까운곳을 찾아서 탐색하고 스택에서는 항상 맨 끝의 노드를 삭제, 삽입, 검색 하므로 연산을 할 위치를 한번에 찾게된다. 따라서 시간복잡도는 `O(1)`이 된다.  
=> 배열을 이용하여 구현하나 연결 리스트를 이용하여 구현하나 모든 연산의 시간복잡도는 `O(1)`이므로 크게 상관없다.  

### 스택의 응용

**수식의 괄호 검사**  
수식에서 괄호과 올바르게 되어 있는지 확인  
`ex) {(1+2)+3}+4 O,  (1+{2+3)+4} X`  

1. 수식을 왼쪽에서 한개씩 읽음
2. 여는 괄호를 만나면 스택에 push
3. 닫는 괄호를 만나면
4. pop을 해서 나오는 괄호가 쌍을 이루는지 확인
5. 쌍을 이루지 않는다면 올바르지 않은 수식
6. 끝까지 진행 후 스택이 비어있다면 올바른 수식  

```python
def solution(expr):
    match = {                   #순서쌍을 dictionary에 저장
        ')': '(',
        '}': '{',
        ']': '['
    }
    S = ArrayStack()            #빈 스택을 생성
    for c in expr:              #수식을 순회
        if c in '({[':          #여는 괄호라면 push
            S.push(c)
        elif c in match:        #여는 괄호가 아니라면
            if S.isEmpty():     #스택이 비어있다면 False를 반환
                return False
            else:    
                t = S.pop()
            if t != match[c]:   #쌍을 이루지 않는다면 False
                return False
    return S.isEmpty()          #마지막에 스택이 비었다면 True
```
<p>&nbsp;</p>  

**수식의 후위 표기법**  
- 중위표기법(infix notaion): 우리가 일반적으로 사용하는 연산자가 피연산자들의 사이에 위치  
`ex) (A+B) * C`  
- 후위표기법(postfix notation): 연산자가 피연산자들의 뒤에 위치  
`ex) A B+ C *`  
=> 괄호 없이 모든 중위 표기법으로 된 수식을 표현할 수 있다.  
<p>&nbsp;</p>  

**중위 표기 수식 -> 후위 표기 수식**  

1. 왼쪽부터 순차적으로 읽음  
읽은 값이  
    - 숫자일겅우  
    그대로 출력  
    - 연산자일 경우  
    스택이 비어있다면 스택에 push해주고  
    아니라면 맨 위에 높거나 같은 우선순위의 연산자들이 안나올때까지 스택을 pop해주고 출력  
    반복 끝에 스택이 비어있거나 맨 위에 우선순위가 낮은 연산자가 있다면 push 해줌  
    - 괄호일 경우  
    여는 괄호라면 스택에 push해줌  
    닫는 괄호라면 여는 괄호가 나올 때까지 스택을 반복적으로 pop해주고 출력
    여는 괄호는 우선순위가 가장 낮도록 설정  
2. 맨끝에 도달했다면 스택에 남아있는 모든 값들을 pop해주고 출력  

정리   
스택을 이용하여 높은 우선순위의 연산자가 먼저 계산되도록 낮은 우선순위의 연산자보다 위에 있게 적절히 pop, push해준다.   
괄호는 닫는 괄호가 나오면 무조건 계산하여 먼저 계산되도록 한다.  

```python
prec = {                                    #연산들의 우선순위들을 dictionary로 저장
    '*': 3, '/': 3,
    '+': 2, '-': 2,
    '(': 1
}

def solution(S):
    opStack = ArrayStack()                  #빈 스택을 생성
    answer = ''
    for val in S:
        if val.isalpha():                   #숫자라면 answer에 바로 추가
            answer += val
        elif val == '(':                    #여는 괄호라면 stack에 push
            opStack.push(val)
        elif val == ')':                    #닫는 괄호라면 여는 괄호가 나올때까지 pop후 answer에 추가
            while opStack.peek() != '(':
                answer += opStack.pop()
            opStack.pop()
        else:                               #연산이라면 맨 위에 낮은 연산자가 나올떄까지 스택을 pop후 answer에 추가
            cur = prec[val]
            while not opStack.isEmpty() and prec[opStack.peek()] >= cur:
                answer += opStack.pop()
            opStack.push(val)
            
    while not opStack.isEmpty():            #스택에 남아있는 연산자들을 answer에 추가
        answer += opStack.pop()
    return answer
```
<p>&nbsp;</p>  

**후위 표기 수식의 계산**  

1. 수식을 왼쪽부터 순차적으로 읽음
2. 숫자가 나오면 push
3. 연산자가 나오면 두 개를 pop해서 연산을 실행 후 결과를 push
(뺄셈과 나눗셈은 순서에 유의해야 한다.)
4. 과정을 마친 후 올바른 수식이라면 스택에 있는 한 개의 값을 출력

```python
def postfixEval(tokenList):                  
    ansStack = ArrayStack()
    for val in tokenList:                       #수식을 왼쪽부터 순회
        if type(val) == int:                    #값이 int라면 stack에 push
            ansStack.push(val)
        else:
            val1 = ansStack.pop()               #연산자라면 숫자 두개를 pop하고 연산을 수행한 후 push
            val2 = ansStack.pop()
            if val == '+':
                ansStack.push(val1+val2)
            elif val == '-':
                ansStack.push(val2 - val1)
            elif val == '*':
                ansStack.push(val1 * val2)
            elif val == '/':
                ansStack.push(val2/val1)
    return ansStack.pop()                       #과정을 마친 후 스택에 마지막 남아있는 값을 반환
```
후위 표기 수식의 계산은 생각보다 코드가 간결한 것을 알 수 있다
컴퓨터로 중위 표기로 된 수식을 계산해야 한다면 후위 표기 수식으로 바꾼 후 계산하는 것이 더 편할 수 있다는 생각이 들었다.
<p>&nbsp;</p>  

## 큐(Queues)  

자료를 보관할 수 있는 선형 구조
- 넣을 때는 한쪽 끝에서 밀어 넣어야 하고 (enqueue)
- 꺼낼 때는 반대 쪽에서 뽑아 꺼내야 한다 (dequeue)
=> FIFO(first in first out)특징을 가지는 선형 자료구조이다.

**연산의 정의**
- `size()`: 큐의 전체 크기를 반환
- `isEmpty()`: 큐가 비어있는지 여부를 반환
- `enqueue(val)`: 큐의 맨 끝에 값을 추가
- `dequeue()`: 큐의 맨 처음 값을 삭제 후 그 값을 반환
- `peek()`: 큐의 맨 처음 값을 반환


### 큐의 구현  
1. 배열을 이용해 구현  
Python 리스트와 메서드들을 이용하여 구현할 수 있다. 하지만 dequeue를 할떄 모든 원소들을 한칸씩 앞으로 당겨줘야하므로 시간복잡도가 `O(n)`으로 효율이 좋지 않다.  

2. 연결 리스트를 이용해 구현  
양방향 연결 리스트를 이용하여 구현 할 수 있다.
=> 양방향 연결 리스트는 삽입, 삭제가 빠르므로 큐를 구현하기에 적합하다.  

```python
class LinkedListQueue:
    def __init__(self):                      
        self.data = DoublyLinkedList()

    def size(self):                               #O(1)
        return self.data.getLength()

    def isEmpty(self):                            #O(1)
        return self.data.getLength() == 0

    def enqueue(self, item):                      #O(1)
        node = Node(item)
        self.data.insertAfter(self.data.tail.prev, node)

    def dequeue(self):                            #O(1)
        return self.data.popAfter(self.data.head)

    def peek(self):                               #O(1)
        return self.data.head.next.data
```
양방향 연결 리스트로 구현하면 모든 연산의 시간복잡도가 `O(1)` 으로 빠른 속도를 가지고 있는 것을 알 수 있다.  



### 환형 큐(Circular Queues)
길이가 정해져있고 처음과 끝이 이어져있는 형태의 큐
- 자료를 성하는 작업과 그 자료를 이용하는 작업이 비동기적으로 일어나는 경우
- 자료를 생성하는 작업이 여러 곳에서 일어나는 경우
- 자료를 이용하는 작업이 여러 곳에서 일어나는 경우
=> 운영체제, CPU 스케줄러 등 컴퓨터 시스템에서 많이 이용된다.  
일정 크기의 배열을 선언하고 실제 사용되는 범위의 시작, 끝점의 위치만 바꿔가는 방식으로 구현하면 배열을 이용해 구현 할 수 있다.  

```python
class CircularQueue:
    def __init__(self, n):                         #크기 n의 list를 가지고 있음
        self.maxCount = n                          #정수형 변수 front, rear를 가지고 있으며 실제 사용되는 범위는 [front+1, rear]이다.
        self.data = [None] * n
        self.count = 0
        self.front = -1
        self.rear = -1
        
    def size(self):
        return self.count

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.maxCount

    def enqueue(self, x):
        if self.isFull():
            raise IndexError('Queue full')  
        self.rear = (self.rear+1)%self.maxCount   #enqueue가 되면 rear를 1 증가시켜주고 maxCount가 되면 0으로 만들어줘서 순환이 되도록 한다.
        self.data[self.rear] = x
        self.count += 1

    def dequeue(self):
        if self.isEmpty():
            raise IndexError('Queue empty')
        self.front = (self.front + 1) % self.maxCount   #dequeue가 되면 front를 1증가시켜주고 maxCount가 되면 0으로 바꿔준다.
        x = self.data[self.front]
        self.count -= 1
        return x

    def peek(self):
        if self.isEmpty():
            raise IndexError('Queue empty')
        return self.data[(self.front+1)%self.maxCount]
```
일반 큐와 마찬가지로 모든 연산의 시간복잡도는 `O(1)`이다.  

### 우선순위 큐(Priority Queues)  
큐가 FIFO방식을 따르지 않고 따로 정의된 원소들의 우선순위에 따라 큐에서 빠져나오는 자료형  

두가지 방식이 가능  
  1. enqueue할 때 우선순위 순서를 유지하도록  
  2. dequeue할 때 우선순위가 높은 것을 선택  

=> 1번 방식이 유리하다!  
우선순위가 높은 원소를 선택하려면 모든 원소를 확인해야 하지만 우선순위를 유지하면서 넣어주는 것은 모든 원소를 확인할 필요는 없다.
우선순위에 따라 정렬시켜 놓고 원소가 들어올 때마다 우선순위가 계속 유지되는 자리에 넣어주면 된다.  

**구현재료**
1. 선형배열  
배열을 정렬 시켜 놓고 새로운 원소가 들어오면 0번 자리에서부터 한칸씩 이동하여 적합한 위치를 찾고 `O(n)` 그 자리에 새로운 원소를 삽입 `O(n)`  
`O(n) + O(n)`  
2. 연결 리스트  
연결 리스트를 정렬 시켜 놓고 새로운 원소가 들어오면 head 노드에서부터 한칸씩 이동하여 적합한 위치를 찾고 `O(n)` 그 자리에 새로운 원소를 삽입 `O(1)`  
`O(n) + O(1)`  

시간: 시간복잡도 자체는 선형배열과 연결 리스트 모두 `O(n)`으로 같지만 연결 리스트가 삽입, 삭제의 시간이 더 빠르므로 연결 리스트가 더 빠르다.  
공간: 공간은 선형배열이 더 조금 차지한다.  
=> 보통 중요한 것은 시간이기 때문에 연결 리스트로 구현하는 것이 더 좋다.

```python
class PriorityQueue:
    def __init__(self):
        self.queue = DoublyLinkedList()

    def size(self):
        return self.queue.getLength()

    def isEmpty(self):
        return self.size() == 0

    def enqueue(self, x):                                           #O(n)
        newNode = Node(x)
        curr = self.queue.head
        while curr.next.next and curr.next.data > x:                #한칸씩 이동하면서 적합한 위치를 찾고 그 위치에 삽입한다.
            curr = curr.next
            self.queue.insertAfter(curr, newNode)
            
    def dequeue(self):                                              #O(1)
        return self.queue.popAt(self.queue.getLength())

    def peek(self):
        return self.queue.getAt(self.queue.getLength()).data
```

연결 리스트로 구현한 우선순위 큐의 시간복잡도는 `enqueue`가 `O(n)` `dequeue`가 `O(1)` 인 것을 확인 할 수 있다.  


## 트리(Trees)  
node와 edge를 이용하여 데이터의 배치형태를 추상화한 자료구조
지금까지 나온 자료구조들과는 다르게 선형자료구조가 아니다

### 용어정리
- **root node**: 트리의 가장 위에 있는 한 개의 노드
- **leaf node**: 더 이상 가지가 없는 가장 아래에 있는 노드
- **internal node**: root도 leaf도 아닌 중간에 있는 노드  
<p>&nbsp;</p>  

- **parent & child (부모, 자식)**: edge로 연결된 두 노드중 root에 가까운게 parent, leaf에 가까운게 child
- **sibling**: 같은 parent를 가지고 있는 노드들
- **ancestor & descendant (조상, 후손)**: 한 노드에서 leaf로 갈 때 마주치는 모든 노드들  
ex) root를 제외한 모든 노드는 root의 descendant이다.
<p>&nbsp;</p>  

- **level**: root로부터 해당 노드로 가기위해 거쳐야할 edge의 수(root는 0)
- **height(depth)**: 모든 노드중 level의 최댓값 +1
- **subtree**: 트리에서 어느 한 노드를 기준으로 후손들을 전부 떼어내서 만든 트리
- **degree**: 특정 노드의 subtree의 개수(child의 개수)  
ex) leaf의 degree는 0이다 (degree가 0이면 leaf이다)  

### 이진트리(Binary Trees)
모든 노드의 degree가 2 이하인 트리  
재귀적 정의: 빈트리 이거나, 루트, 왼쪽 서브트리, 오른쪽 서브트리로 구성되어 있을 때 서브트리들이 이진트리면 이진트리다.

- **Full Binary Tree**: 모든 level에서 노드들이 모두 채워져 있는 이진트리  
(높이가 k이면 노드의 개수는 2^k-1개 이다)  

- **Complete Binary Tree**: 높이 k라면 level이 k-2까지의 모든 노드들은 2개의 자식을 가진 full binary tree이고 k-1부터는 왼쪽부터 순차적으로 채워져 있는 이진트리

**연산의 정의**
- `size()`: 트리에 있는 노드들의 총 개수를 반환
- `depth()`: 트리의 depth(height)를 반환
- `traversal()`: 트리의 모든 노드들을 순회

### 이진트리의 구현  

**Node**  

```python
class Node:
    def __init__(self, item):
        self.data = item
        self.left = None
        self.right = None
```
연결 리스트 때와 마찬가지로 노드를 직접 구현하여 자료구조를 만든다.  
각 노드는 값을 가지고잇는 data, 왼쪽 자식이 들어있는 left, 오른쪽 자식이 들어있는 right 3가지 정보를 가지고 있다.

연산을 구현할 때는 이진트리 클래스에서 직접 구현하지 않고 노드의 클래스에서 미리 구현해 놓고 호출하는 방식으로 구현하였다.  

- `size()`  

```python
def size(self):
    l = self.left.size() if self.left else 0        #왼쪽 노드의 총 개수
    r = self.right.size() if self.right else 0      #오른쪽 노드의 총 개수
    return l + r + 1                                #왼쪽 + 오른쪽 + 1(본인)
```
<p>&nbsp;</p>  

- `depth()`  

```python
def depth(self):
    l = self.left.depth() if self.left else 0       #왼쪽 노드의 depth
    r = self.right.depth() if self.right else 0     #오른쪽 노드의 depth
    return max(l,r) + 1                             #둘 중 최대값 + 1
```
노드에서 구현된 위의 함수들을 이진트리 클래스에서 호출만 하면 값을 얻을 수 있다.  

**이진트리의 순회(Traversal)**

1. 깊이 우선 순회(Depth First Traversal)
  - 중위 순회(in-order traversal): left->root->right
  - 전위 순회(pre-order traversal): root->left->right
  - 후위 순회(post-order traversql): left->right->root  

=>재귀 함수로 구현  

```python
class Node:
    def inorder(self):
            traversal = []
            if self.left:
                traversal += self.left.inorder()
            traversal.append(self.data)
            if self.right:
                traversal += self.right.inorder()
            return traversal

    def preorder(self):
        traversal = []
        traversal.append(self.data)
        if self.left:
            traversal += self.left.preorder()
        if self.right:
            traversal += self.right.preorder()
        return traversal

    def postorder(self):
            traversal = []
            if self.left:
                traversal += self.left.postorder()
            if self.right:
                traversal += self.right.postorder()
            traversal.append(self.data)
            return traversal
```
구현한 내용을 보면 왼쪽을 탐색하는 부분, 오른쪽을 탐색하는 부분, 본인을 확인 하는 부분의 순서만 바뀌고 내용은 동일한 것을 알 수 있다.  

2. 넓이 우선 순회(Breadth First Traversal)  

순회 순서  
- level이 낮은 노드를 우선으로 방문
- 같은 level 노드들 사이에서는 부모 노드의 방문 순서에 따라 방문
- 왼쪽 자식노드를 오른쪽보다 먼저 방문

방문했던 순서를 기억해야하기 때문에 시행마다 새롭게 순회하는 재귀적 방법은 적합하지 않다.  
=> 방문했던 순서대로 다시 순회할 수 있도록 Queue를 이용하자!

```python
class BinaryTree:
    def __init__(self, r):
        self.root = r
    def bft(self):
        if not self.root:
            return []
        traversal = []
        bftQueue = ArrayQueue()
        bftQueue.enqueue(self.root)                 #처음에는 큐에 root를 저장
        while not bftQueue.isEmpty():               #큐가 비어있을 때까지 반복
            node = bftQueue.dequeue()               #큐에서 dequeue해서 저장
            traversal.append(node.data)
            if node.left:                           #저장된 노드의 왼쪽, 오른쪽 노드들을 있다면 큐에 enqueue
                bftQueue.enqueue(node.left)
            if node.right:
                bftQueue.enqueue(node.right)
        return traversal
```
깊의 우선 순회와 넓이 우선 순회는 순회하는 노드의 순서만 다를 뿐 시간복잡도는 동일하다.  

### 이진 탐색 트리(Binary Search Trees)  
모든 노드에 대해서  
- 왼쪽 서브트리에 있는 데이터는 모두 현재 노드의 값보다 작고
- 오른쪽 서브트리에 있는 데이터는 모두 현재 노드의 값보다 큰  

성질을 만족하는 이진트리 (중복되는 데이터의 원소는 없는 것으로 가정)  
추상적 자료구조로 각 노드는 (key, value)의 쌍으로 표현한다. key는 중복 될 수 없는 숫자값이고, value는 해당 key가 가지고 있는 정보다.  

탐색의 최소 시간복잡도는 `O(log n)`으로 빠른 탐색 속도를 가지고 있으며 같은 탐색속도를 가지고 있는 정렬된 배열에서의 이진 탐색하는 방법보다 원소의 삽입, 삭제가 더 빠르다.

**연산의 정의**
- `insert(key, data)`: 원소 추가
- `remove(key)`: 해당 key 삭제
- `lookup(key)`: 해당 key 탐색
- `inorder()`: key의 순서대로 데이터 원소를 나열
- `min()`, `max()`: 최소 key, 최대 key를 가지는 원소를 탐색

### 이진 탐색 트리의 구현 

- `insert(key, data)`  

root부터 비교하면서 자신보다 큰 값이면 오른쪽으로 작으면 왼쪽으로 이동한다.
빈 자리가 나올 때까지 반복하며 빈자리가 나오면 그 자리에 삽입한다.  

이번에도 Node class에서 구현하고 이진 탐색 트리에서는 호출만 하는 것으로 구현했다.
```python
class Node:
    def __init__(self, key, data):              #각 노드들은 key, data, left, right 4가지 정보를 가지고 있음
        self.key = key
        self.data = data
        self.left = None
        self.right = None

    def insert(self, key, data):
        if self.key < key:                        #새로운 노드가 더 크면 오른쪽 노드로 이동
            if self.right:
                self.right.insert(key, data)
            else:                                 #노드가 비어있다면 그 자리에 삽입 후 종료 
                self.right = Node(key, data)
        elif self.key > key:                      #새로운 노드가 더 작으면 왼쪽 노드로 이동
            if self.left:
                self.left.insert(key, data)
            else:                                 #노드가 비어있다면 그 자리에 삽입 후 종료
                self.left = Node(key, data)
        else:                                     #같은 값이 존재한다면 에러
            raise KeyError('exist key')
```
코드를 보면 한번 진행할 때마다 트리에서 한 칸씩 내려가므로 최대 트리의 depth만큼 시행되고 최소시간복잡도는 `O(log n)` 이다.  

- `lookup(key)`  

```python
class Node:
    def lookup(self, key, parent=None):
            if key < self.key:
                if self.left:
                    return self.left.lookup(key, self)
                else:
                    return None, None
            elif key > self.key:
                if self.right:
                    return self.right.lookup(key, self)
                else:
                    return None, None
            else:
                return self, parent
```
`insert`와 같은 방법으로 구현하였고 아래에서 구현할 `remove`에서 부모의 정보도 필요하기 때문에 찾은 노드와 부모의 노드를 반환하도록 구현하였다. 시간복잡도는 `insert`와 같다.


- `remove(key)`  
노드의 삭제는 삭제된 노드의 자식을 처리하는 부분이 복잡하기 때문에 경우를 나누어서 구현한다.

1. 삭제되는 노드가 leaf인 경우  
노드를 삭제한 후 부모 노드의 링크만 조정해주면된다.  
(부모노드에서 삭제되는 노드의 링크를 None으로 바꿔주어야 한다.)  

2. 삭제되는 노드가 자식을 하나 가지고 있는 경우  
삭제되는 노드 자리에 그 자식을 대신 배치한다.  
-> 자식이 오른쪽인지 왼쪽인지 확인 후  그 노드를 배치한다.  
-> 부모노드의 링크를 조정해준다.(1번과 동일)  

3. 삭제되는 노드가 자식을 둘 가지고 있는 경우
삭제되는 노드보다 큰 노드들중 가장 작은 노드를 찾아서 삭제되는 노드의 위치에 교체해 주고 그 노드를 삭제한다.  
(삭제되는 노드보다 작은 가장 큰 노드로 해도 가능하다.)  
삭제과정: 찾은 노드의 자식은 없거나 오른쪽자식 한개만 있는 경우 밖에 없음로 1,2번 경우와 동일하게 처리해 주면 된다.

이번에는 이진 탐색 트리 class에 직접 구현했다.

```python
class Node:                       #remove에서 사용하기위해 만든 자식의 개수를 세주는 함수
  def countChildren(self):
      count = 0
      if self.left:
          count += 1
      if self.right:
          count += 1
      return count
      
class BinSearchTree:
    def remove(self, key):
        node, parent = self.lookup(key)         #삭제할 노드와 그 부모를 찾아서 저장한다.
        if node:
            nChildren = node.countChildren()
            if nChildren == 0:                  #자식이 0개라면(leaf라면)
                if parent:                      #부모에게서 어디에 연결되어 있는지 찾은 후 링크를 삭제한다.
                    if parent.left == node:
                        parent.left = None
                    else:
                        parent.right = None
                else:                           #부모가 없다면 root이므로 root를 삭제한다.
                    self.root = None
            elif nChildren == 1:                #자식이 1개라면
                child = node.left if node.left else node.right    #1개인 자식을 찾아서 저장한다.
                if parent:                      #부모에게서 어디에 연결되어 있는지 찾은 후 자식을 연결시켜준다.
                    if parent.left == node:
                        parent.left = child
                    else:
                        parent.right = child
                else:
                    self.root = child
            else:                               #자식이 2개라면
                parent = node
                successor = node.right          #트리에서 오른쪽으로 한번 움직인 후 왼쪽이 없을 때까지 왼쪽으로 이동한다.
                while successor.left:           #삭제할 노드보다 한단계 큰 노드를 찾아서 successor에 저장하고 그 부모를 parent에 저장한다.
                    parent = successor
                    successor = successor.left
                    
                node.key = successor.key        #삭제할 노드의 값을 successor의 값들로 바꿔준다.
                node.data = successor.data
                                                                        #successor는 왼쪽으로 최대한 이동했기 때문에 자식이 오른쪽에 있거나 없거나 둘중 하나이다.
                child = successor.right if successor.right else None    #successor의 자식을 child에 저장 후  parent의 적절한 위치에 연결시켜준다.
                if parent.left == successor:
                    parent.left = child
                else:
                    parent.right = child
            return True
```
코드를 보면 상당히 길고 복잡해 보이지만 최소 시간복잡도는 여기서 사용한 `lookup`의 시간 복잡도인 `O(log n)`이다. 하지만 노드를 찾는 과정을 여러번 수행하므로 삽입보다는 오래걸릴 것이라고 추측해 볼 수 있다.  

**이진 탐색 트리가 효율적이지 못할 경우**  
위에서 시간복잡도를 설명할 때 지금까지와는 달리 항상 **최소**시간복잡도라고 설명했다. 
그 이유는 특정 상황에서 시간복잡도가 `O(n)`까지 떨어질 수 있기 때문인데 그 경우는 원소를 크기순서대로 삽입할 경우이다.  
이 경우에는 트리가 일렬로 생성되어 연결리스트와 비슷한 모양을 가지게 된다.  
이진 탐색 트리가 효율적이기 위해서는 오른쪽과 왼쪽의 노드가 동등하게 배치될 수록 좋아지게 되며 완벽히 균형을 이루었다면 `O(log n)`의 시간 복잡도가 나오게 된다.  
높이와 균형을 유지하는 이진 탐색 트리도 있다.(AVL tree, Red-black tree)  
이 경우에는 항상 `O(log n)`의 탐색 복잡도를 가지고 있지만 삽입, 삭제 연산이 복잡하다.  

## 힙(Heaps)  
따로 분류를 해놓긴 했지만 힙은 이진트리의 한 종류이며 다음과 같은 조건을 만족한다.
- root가 언제나 최댓값 또는 최솟값을 가진다. (각 경우마다 최대힙(max heap), 최소힙(min heap)이라고 부른다.)  
- 완전 이진 트리(Complete Binary Trees) 여야 한다.  

이진 탐색 트리와의 비교
- 원소들이 완전히 크기 순으로 정렬되어 있는가: X
- 특정 키 값을 가지는 원소를 빠르게 검색할 수 있는가: X
- 부가의 제약조건은 어떤것인가: 완전 이진 트리여야 함  

**연산의 정의**
- `insert(val)`: 새로운 원소를 삽입
- `remove()`: 최대원소(root)를 반환과 동시에 

### 힙의 구현  

완전 이진 트리라면 배열로 트리를 표현할 수 있다.  
힙은 완전 이진 트리므로 배열로 구현할 수 있으며 배열로 구현하면 공간도 아낄 수 있고 간단하게 구현할 수 있다.  
배열에 끝에 원소가 추가될 때마다 트리에 같은 level에서 왼쪽에서 오른쪽으로 값이 들어간다고 생각하면된다.  
완전 이진 트리는 빈 공간 없이 채워지고 삭제되므로 원소의 추가와 삭제가 항상 끝에서 일어나기 때문에 가능하다.  

그렇게 구현하면 노드번호가 m이라면  
`왼쪽 자식의 번호: 2 * m`  
`오른쪽 자식의 번호: 2 * m + 1`  
`부모 노드의 번호: m // 2`  

와 같이 생각 하면 완전 이진 트리와 배열이 일대일 대응이 된다. (0번 노드는 비워놓는다.)  


- `insert(val)`  

1. 트리의 마지막 자리에 새로운 원소를 임시로 저장 (배열의 맨 끝)
2. 새로운 원소에서 시작해서 부모 노드와 키 값을 비교하여 부모보다 더 크다면 swap. 최대힙의 조건을 만족할 때까지 반복

파이썬 팁  
`a, b = b, a`을 사용하면 한줄로 원소를 교체할 수 있다.

```python
class MaxHeap:
    def __init__(self):
        self.data = [None]
    def insert(self, item):
        self.data.append(item)
        index = len(self.data)-1
        while index > 1 and self.data[index//2] < self.data[index]:    #부모노드가 더 작을때까지 부모와 자식을 바꿔줌
            self.data[index//2] , self.data[index] = self.data[index], self.data[index//2]
            index = index//2
```
`insert`의 시간복잡도는 한번에 트리의 level의 한칸씩 움직이고 최대 depth만큼 움직일 수 있기 때문에 depth의 값인 `O(log n)`가 된다.  

- `remove()`  

과정  
1. root제거
2. 트리의 마지막에 있는 노드를 root가 있던 자리에 배치
3. 자식 노드들과 키 값을 비교해서 자식이 더 크다면 swap. 최대힙의 조건을 만족할 때까지 반복
4. 삽입때와는 다르게 부모는 자식이 두개가 있을 수 있는데 둘 중 더 큰 값을 선택하여 swap해주면 된다.

```python  
def remove(self, i):
        left = 2 * i             #자식들의 인덱스를 계산
        right = 2 * i + 1
        m = i             #자식들중 큰 값을 선택해야 하기 때문에 비교할 변수 선언
        if left < len(self.data) and self.data[left] > self.data[i]:
            m = left      #왼쪽 자식이 더 크다면 값을 저장
        if right < len(self.data) and self.data[right] > self.data[m]:
            m = right     #오른쪽 자식이 저장된 값보다 더 크다면 값을 저장
        if m != i:        #값이 그대로가 아니라면 현재값이 최대가 아니라는 뜻이므로 swap해주고 다시 실행
            self.data[m], self.data[i] = self.data[i], self.data[m]
            self.maxHeapify(m)

```
`remove`의 시간복잡도는 최대값을 찾는 과정이 추가로 있긴 하지만 `insert`와 같은 `O(log n)`인 것을 알 수 있다.  

### 힙의 응용  

- 우선순위 큐(Priority Queues)  
힙으로 우선순위 큐를 구현하면 전에 연결 리스트로 구현했을 때보다 더 효율적으로 구현 할 수 있다.  

  |      |enqueue        |dequeue  |
  |:--- | :---: | :---: |
  | 연결 리스트 | O(n)     | O(1)     |
  | 우선순위 큐 | O(log n) | O(log n) |
  
- 힙 정렬(heap sort)  
정렬되지 않은 원소들을 아무 순서로 최대 힙에 삽입  
삽입이 끝나면 힙이 비게 될 때까지 하나씩 삭제  
=> 시간복잡도: `O(log n)`을 n번 반복하므로 `O(n log n)` (merge sort와 동일)  





 
