---
title: "AI 노트"
toc: true
toc_sticky: true
date: 2020-11-29
categories: 노트
use_math: true
---

**필요해 보이는 것들 계속 정리**  

## numpy
느린 python대신 c로 미리 구현을 해서 빠른 속도를 가지게한 것
import numpy as np  

### 함수
**np.array(list)**  
list를 array로 변환  

<p>&nbsp;</p>  

**arr.shape**  
array의 모양(차원)을 알려줌  
<p>&nbsp;</p>  

**np.matmul(A,B)**  
A, B의 행렬곱을 반환  
동일 표현 np.dot(A,B), A.Matmul(B), A.dot(B)  
둘다 array라면 A @ B 로도 표현 가능  
<p>&nbsp;</p>  

**np.zeros(dim)**  
dim의 영행렬 반환  
np.ones(dim)도 똑같이 됨  
<p>&nbsp;</p>  

**np.diag((a,b,c))**  
diagonal matrix 생성  
<p>&nbsp;</p>  

**np.eye(n, dtype = float)**  
nxn float타입의 항등행렬 생성  
dtype생략시 float 위에거들도 다 dtype 사용가능  
<p>&nbsp;</p>  

**np.trace(A)**  
main diagonal의 합 반환  
A가 array면 A.trace()로도 가능  
<p>&nbsp;</p>  

**np.linalg.det(A)**  
determinant반환  
<p>&nbsp;</p>  

**np.linalg.inv(A)**  
역행렬 반환  
<p>&nbsp;</p>  

**np.linalg.eig(A)**  
고유값과 고유벡터를 튜플로 묶어서 반환  
<p>&nbsp;</p>  

**np.arange(start, end, diff)**  
start부터 end전까지 (포함안함) diff간격으로 array만듬  
np.arange(n): 0~n-1까지 정수로 만듬  
<p>&nbsp;</p>  


### 지식  

**numpy에서는 벡터를 쓸때 거의 행벡터를 사용하며 열벡터가 들어갈 자리에 행벡터를 사용해도  
행렬 모양을 보고 알아서 열벡터로 바꿔서 연산을 해준다.**  

**list[i][j] = array[i,j]**  
array[:][a:b]  

**Broadcasting**  
차원이 다른 행렬끼리 연산해도 연사이 가능하도록 변환이 되면 복사한다음에 연산된다  

## pandas  
table의 역할을 대체하기 위해 많이 사용  
import pandas as pd  

### Series
**s = pd.Series(list)**  
list를 series로 변환  
<p>&nbsp;</p>  

**t = pd.Series(dic)**  
dictionary를 series로 변환  
<p>&nbsp;</p>  

**s[s>s.median()]**  
특정조건을 만족하는 원소만 추출  
<p>&nbsp;</p>  

**s[[3,1,4]]**  
해당 인덱스만 추출  
<p>&nbsp;</p>  

**s.dtype**  
데이터 타입 반환  
<p>&nbsp;</p>  

**dic과 사용법이 동일**  
t[key]   
t[newkey] = nval  
t.get(key, 0)  
<p>&nbsp;</p>  

**name property**  
s = pd.Series(np.random.randn(5), name = "random_nums")  
<p>&nbsp;</p>  

**s.value_counts()**  
series에서 데이터의 항목별 개수를 세줌  
<p>&nbsp;</p>  

### DataFrame
**df = pd.DataFrame(dic)**  
dictionary를 dataframe으로 변환  
df = pd.DataFrame(d, index = list}  
df.dtypes  
<p>&nbsp;</p>  

**df = pd.read_csv(path)**  
csv파일을 dataframedmfh 변환  
<p>&nbsp;</p>  

**df.head(n)**  
처음 n개의 데이터 참조  

**df.tail(n)**  
마지막 n개의 데이터 참조  
<p>&nbsp;</p>  

**df[‘column_name’] or df.column_name**  
해당 열 접근, 후자의 접근방법은 띄어쓰기 같은걸 인식을 못함  
<p>&nbsp;</p>  

**df[['a', 'b']]**  
리스트로 넣으면 여러개의 열을 추출할 수 있다.  

**covid[covid[‘New cases’] > 100]**  
해당 조건을 만족하 row만 추출  
<p>&nbsp;</p>  

**covid[‘WHO Region’].unique()**  
데이터들을 중복 없이 출력, 범주의 종류를 확인  
<p>&nbsp;</p>  

**df.loc[row_name]**  
해당 row 추출
df.loc[:, col_name]: 해당 col 추출  

**df.loc[row_name, col_name]**  
해당 row, col에 해당하는 값 반환  
df.loc[row_name][col_name] 해도됨  

**df.iloc[rowidx, colidx]**  
numpy와 완전히 동일 slicing가능  
<p>&nbsp;</p>  

**groupby**  
covid_by_region = covid[‘Confirmed’].groupby(by=covid[“WHO Region”])  
.groupby('WHO Region') 해도됨  

.groupby(['a','b'])와 같이 여러기준으로 그룹도 가능  

covid_by_region.sum()  
지역별 데이터의 합을 출력해 준다.  

covid_by_region.mean()  
데이터들의 평균을 출력해 준다.  
<p>&nbsp;</p>  

**df.corr()**  
상관계수를 행렬형식으로 나타내준다.  
<p>&nbsp;</p>  

**df.describe()**  
수치형 데이터 전체의 요약을 알려준다. (평균 등)  
<p>&nbsp;</p>  

**df.isnull()**  
null인 항목, 결측치를 알려줌  
.sum()을 붙이면 더 보기 편함  
<p>&nbsp;</p>  



## matplotlib (+ seaborn)  

### matplotlib  
import matplotlib.pyplot as plt  
#matplotlib inline : 시각화를 주피터 노트북의 incell에서 진행  
<p>&nbsp;</p>  


**plt.figure(figsize = (6,6))**
맨처음에 실행  
그림이 그려질 사이즈 조절가능  
<p>&nbsp;</p>  

**plt.subplot(가로, 세로, figsize = (10,5))**  
fig, ax = plt.subplot(a, b, figsize = (10,5))  
그림 여러개를 한번에 그릴 때 사용  
가로 a개 세로 b개 그리고 fix, ax를 추가로 이용  
<p>&nbsp;</p>  

**label달기**  
plt.xlabel(“x value”)  
plt.ylabel(“f(x) value”)  
<p>&nbsp;</p>  


**plt.axis([x_min, x_max, y_min, y_max])**  
그래프의 보고싶은 범위를 설정  
<p>&nbsp;</p>  

**눈금설정**  
plt.xticks([i for i in range(-5, 6, 1)])  
plt.yticks([i for i in range(0, 27, 3)]  
<p>&nbsp;</p>  

**plt.title(”title name”)**  
제목 달아줌  
<p>&nbsp;</p>  

**plt.plot(x, y, label = "label name")**  
꺾은선 그래프를 그려줌  
x를 생략하면 인덱스가 x가 됨  
<p>&nbsp;</p>  


**plt.legend()**  
범주생성, label을 설정하고 plot뒤에 나와야 함  
<p>&nbsp;</p>  

**plt.show()**  
그래프를 그려줌  
<p>&nbsp;</p>  

**그래프 그리는 종류**  

**plt.scatter(x,y)**  
산점도 (Scatter Plot), 점으로 표현  
<p>&nbsp;</p>  

**plt.boxplot(y)**  
박스그림 (box plot)  
수치형 데이터에 대한 정보 (Q1, Q2, Q3, min, max)  
<p>&nbsp;</p>  

**plt.bar(x, y)**  
plt.bar(x = X, height = Y)  
막대 그래프(bar plot)  
범주형 데이터의 값과 그 값의 크기를 직사각형으로 나타낸 그림  
<p>&nbsp;</p>  

**plt.hist(y, bins = np.arange(0,20,2))**  
히스토그램   
연속적인 막대그래프  
<p>&nbsp;</p>  

**plt.pie([1,2,3,4], labels = [‘a’,’b’,’c’,’d’])**  
원형 그래프 (pie chart)  
데이터에서 전체에 대한 부분의 비율을 부채꼴로 나타낸 그래프  
<p>&nbsp;</p>  


### Seaborn

 import seaborn as sns  
**s.fig.set_size_inches(10, 6)**  
사이즈 조절  
<p>&nbsp;</p>  


x = np.arange(0,22,2)  
y = np.random.randint(0,20,20)   
위와 같은 형식으로 입력  

**sns.kdeplot(y, shade = True)**  
커널밀도그림 (Kernel Density Plot)  
히스토그램과 같은 연속적인 분포를 곡선화해서 그린 그림  
shade를 True로 해주면 색칠해줌  
y 축은 density  
<p>&nbsp;</p>  

**sns.countplot(x = vote_df[‘group’])**  
sns.countplot(x = 'group', data = vote_df)랑 똑같음  
카운트그림 (Count Plot) 막대그래프  
범주형 column의 빈도수를 시각화 -> Groupby 후의 도수를 하는 것과 동일한 효과  
<p>&nbsp;</p>  


**sns.catplot()**  
s = sns.catplot(x = ’WHO Region’, y = ‘Confirmed’, data  = covid, kind = ‘strip’)  
캣 그림 (cat plot)  
숫자형 변수와 하나 이상의 범주형 변수의 관계를 보여주는 함수  
x: 범주형변수, y: 숫자형 변수  
, hue: 추가범주, : hue로 범주를 추가 할 수 있다.  
<p>&nbsp;</p>  

**sns.stripplot(x = ‘WHO Region’, y = ‘Recovered’ , data = covid)**  
스트립그림 (strip plot)  
scatter plot과 유사하게 데이터의 수치를 표현하는 그래프  
범주별로 표현이 가능하다  
<p>&nbsp;</p>  

**sns.swarmplot(x = ‘WHO Region’, y = ‘Recovered’ , data = covid)**  
똑같은데 중복값을 더 잘보여줌  
<p>&nbsp;</p>  

**sns.heatmap(covid.corr())**  
히트맵 (Heatmap)  
데이터의 행렬을 색상으로 표현해주는 그래프  
<p>&nbsp;</p>  



## git (+cli command)

### git  
git init: 현재 디렉토리를 git 저장소로 지정  

git add 파일명: 해당 파일을 staged상태로 만듬  
git commit -m "message": 메세지와 함께 커밋  
git push orign master: origin에 master 브랜치로 push  

git reset HEAD 파일명: add 취소  
git reset HEAD~: 커밋 취소 (add도 취소됨)  

git log: 커밋한 기록들 확인  
git status: unstaged된 파일에 대한 정보가 나옴, 새로 커밋 될 수 있는 변경사항에 대해 나옴  

git branch -v: 현재 branch 상태 확인  
git branch branch_name: branch 생성  
git checkout branch_name: 작업환경을 입력한 branch로 변경  
git merge branch_name: 현재 branch와 입력한 branch를 병합  
git branch -d branch_name: 입력한 branch를 삭제  

git remote add 별칭 주소: 원격저장소 등록 후 별칭 지정  
git remote -v: 저장되어 있는 원격 저장소 확인  
git branch -M branch_name: branch의 이름을 변경  

git clone 주소 폴더이름: 폴더가 생기고 복사한 내용을 넣어줌  
git clone -b branch_name 주소: 한 개의 branch만 clone  

**fetch 후 merge하는 방법**  

git fetch corini  
git merge corini/week4  
fetch를 해오면 해당 주소의 모든 branch에 대해 복사한 후 branch로 가져온다.  
가져온 branch와 내 branch를 병합하면 pull과 같은 결과가 된다.  

git worktree add -b branchname path: branch들을 독립적인 폴더에서 관리 할 수 있다.  



### cli  
ls: 파일탐색  
ls -al: 모든 파일 탐색  
cd ..: 상위파일 이동  
cat 파일명: 파일 내용 확인  
vim 파일명: 파일 편집 (없다면 생성)
vim 사용법    
i - 편집  
:q - 저장안하고 나가기  
:wq - 저장하고 나가기  





## Jupyter NoteBook

terminal  
jupyter notebook  

jupyter nbconvert --to markdown notebook.ipynb: ipnb파일을 markdown으로 변환  


### 단축키  

#### 입력모드 (enter)
- shift + enter: cell 실행 후 다음 cell 생성
- ctrl + enter: cell 실행

#### 명령모드 (esc)  
- m: 마크다운 입력 모드  
- y: 파이썬 입력 모드  
- a: 현재 cell 위에 새로운 cell 추가
- b: 현재 cell 아래에 새로운 cell 추가
- dd: 현재 cell 삭제


## WEB  

### django

**프로젝트 만들기**  
django-admin startproject projname  
<p>&nbsp;</p>  

**admin 계정 생성**  
python manage.py migrate  
python manage.py createsuperuser  
<p>&nbsp;</p>  

**app만들기**  
django-admin startapp appname  
프로젝트 경로 안에서 실행  
<p>&nbsp;</p>  


**저장**  
python manage.py makemigrations homepage  
git add와 비슷한 것으로 migrate할 항목을 업데이트 한다.  

python manage.py migrate  
를 해줘야 클래스 형태의 모델이 연동되어 admin에서 사용할 수 있다.  

git add -> git commit과 비슷한 작업  
<p>&nbsp;</p>  

**csrf무시**
from django.views.decorators.csrf import csrf_exempt  
from django.utils.decorators import method_decorator  

<p>&nbsp;</p>  

render(request, 'app.html', dict): 해당 페이지 render  


**db**  
Model.objects.all()  
Model.objects.get(id = pk)  

form = Form()  
form.is_valid()  
form.save  
<p>&nbsp;</p>  

**request**  
<input type="hidden" name = "_method" value = "delete">  
request.POST.get('key', '')  
request.POST['key']  
<p>&nbsp;</p>  


### cli  

virtualenv venv:  가상환경 생성  

가상환경 진입  
mac: source venv/bin/activate  
win: ./venv/Scripts/activate.bat  

deactivate: 비활성화  
pip freeze: 설치된 모듈 전부 보여줌  



## 통계

**numpy**  
np.var(list, ddof=1)  
np.std(list, ddof=1)   
ddof=1은 표본이라는 뜻  

np.max()  
np.min()  
np.quantile(list, 0.25)  

np.random.exponetial(scale = 3, size = n) 지수분포에서 랜덤추출  
np.random.rand(n): 0~1사이의 값 n개 (uniform)  
np.random.randn(n): 정규분포(평균:0, 분산: 1)  
np.random.randint(0,20,20): 0~20(미포함)까지의 정수를 20개 만듬  


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

scipy.stats.norm.ppf(1-$\alpha$/2) : z값을 구해준다. 

**statistics**  
import statistics  

statistics.mean()  
statistics.variance()  
statistics.pvariance()  
statistics.stdev()  
statistics.pstdev()   
p가 붙어있는게 모분산, 모표준편차  



















