---
title: "AI 노트"
toc: true
toc_sticky: true
date: 2020-12-07
categories: 노트
---

**필요해 보이는 것들 계속 정리**  

## numpy

import numpy as np  

### 함수
- **np.array(list)**  
list를 array로 변환  

- **arr.shape**  
array의 모양(차원)을 알려줌  

- **np.matmul(A,B)**  
A, B의 행렬곱을 반환  
둘다 array라면 A @ B 로도 표현 가능  





## Jupyter NoteBook

terminal  
jupyter notebook  

### 단축키  

#### 입력모드 (enter)
- shift + enter: cell 실행 후 다음 cell 생성
- ctrl + enter: cell 실행

#### 명령모드 (esc)  
- m: 마크다운 입력 모드  
- y: 파이썬 입력 모드  
- a: 현재 cell 위에 새로운 cell 추가
- b: 현재 cell 아래에 새로운 cell 추가


## 통계

**numpy**  
- np.var(list, ddof=1)
- np.std(list, ddof=1)  
ddof=1은 표본이라는 뜻  
- np.max()
- np.min()
- np.quantile(list, 0.25)


**scipy**  
import scipy  
import scipy.stats  
- scipy.stats.tvar(a)
- scipy.stats.zscore(a, ddos=1)  
ddos는 표본일 때

from scipy import stats

이항분포  
1 - stats.binom.cdf(0, n=3, p=0.2)  
stats.binom.stats(n=3, p=0.2): 평균, 분산 출력  

정규분포  
stats.norm.cdf(4, loc=4, scale=3)  
stats.norm.cdf(7, loc=4, scale=3) - stats.norm.cdf(4, loc=4, scale=3)  
loc = $\mu$, scale = $\sigma$  

포아송분포  
stats.poisson.cdf(2, mu=3)  

지수분포  
lambda = 3
stats.expon.cdf(0.5, scale = 1/lambda)  

**statistics**  
import statistics  

- statistics.mean()
- statistics.variance()
- statistics.pvariance()
- statistics.stdev()
- statistics.pstdev()  
p가 붙어있는게 모분산, 모표준편차



















