---
title: "[인공지능 데브코스] 3주차 day4 - EDA"
toc: true
toc_sticky: true
date: 2020-12-17
categories: TIL
---

## 12월 17일 목   

오늘은 데이터를 분석하는 방법중 하나인 EDA에 대해서 공부했다. EDA의 과정에 대해서 배우고 주피터 노트북의 환경에서 numpy, pandas, matlibplot 을 이용하여 다양한 방식으로 데이터를 분석해 보았다.  


## EDA란?   
데이터 자체의 특성을 육안으로 확인하는 과정  

데이터를 분석하는 기술적 접근은 매우 많다.  
**데이터 그 자체**만으로부터 인사이트를 얻어내는 접근법  


## EDA의 process  

1. 분석의 목적과 변수 확인
2. 데이터 전체적으로 살펴보기 
3. 데이터의 개별 속성 파악하기



## EDA with Example - Titanic

### 분석의 목적과 변수 확인  
살아남은 사람들은 어떤 특징을 가지고 있었을까?   
titanic_df.dtypes : 타입 확인  
<p>&nbsp;</p>  

### 데이터 전체적으로 살펴보기  
titanic_df.describe()  
수치형 데이터에 대한 요약 제공  

titanic_df.corr()  
상관계수 확인  
Correlation is Not Causation  
상관성: 경향성만 나타냄  
인과성: a가 일어나면 b가 일어난다  

상관성이 있다고 반드시 인과성이 있다고는 할 수 없다.  

titanic_df.isnull().sum()  
null인 항목을 찾아줌 (결측치를 찾아줌)  
결측치를 처리할 방법이 많기 때문에 어떻게 결정할지 선택해야함  
<p>&nbsp;</p>  


### 데이터의 개별 속성 파악하기 

**survived column**  
titanic_df[‘Survived’].sum()  
titnaic_df[’Survived’].value_counts()  
sns.countplot(x = ‘Survived’, data =  titanic_df)  
<p>&nbsp;</p>  

**Pclass**  
titanic_df[[‘Pclass’, ‘Survived’]].goupby(‘Pclass’)  
titanic_df[[‘Pclass’, ‘Survived’]].goupby(‘Pclass’).mean() : 생존률  
<p>&nbsp;</p>  

**Sex**  
titanic_df.groupby([‘Survived’, ‘Sex’])[’Survived’].count()  
sns.catplot(x = ‘Sex’, col = ‘Suvived’, kind = ‘count’, data = titanic_df)  
<p>&nbsp;</p>  

**Age**  
remind : 결측치 존재  

fig, ax = plt.subplots(1,1, figsize = (10,5))  
sns.kdeplot(titanic_df[titanic_df.Survived == 1]['Age'], ax = ax)  
sns.kdeplot(titanic_df[titanic_df.Survived == 0]['Age'], ax = ax)  
plt.legend(['Survived', 'Dead'])  
plt.show()  
<p>&nbsp;</p>  

**Sex + Pclass vs Survived**  
sns.catplot(x = ‘Pclass’, y = ’Survived’, hue = ’Sex’,kind = ‘point’, data = titanic_df)  
hue로 추가 변수에 대해 파악  
<p>&nbsp;</p>  

**Age + Pclass**  
titanic_df[‘Age’][titanic_df.Pclass == 1].plot(kind=‘kde’)  

<p>&nbsp;</p>  