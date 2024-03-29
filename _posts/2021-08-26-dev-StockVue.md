---
title: "[개발] Stock Price Prediction - Flask, Vuejs"
toc: true
toc_sticky: true
date: 2021-08-26
categories: 개발 web Flask vue
---

**주식 가격 예측 웹 프로젝트**

&nbsp;

이전에 [time series forecasting 프로젝트](https://doheon.github.io/%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90/time-series/ci-6.compare-post/) 를 통해 5가지의 방법으로 동일한 시계열 데이터에 대해 예측을 진행하고 성능을 비교했었다.

그 중에서 가장 성능이 좋았던 [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/pdf/2012.07436v3.pdf) 라는 논문의 방법을 사용해서 시계열 예측의 대표적인 task인 주식가격 예측을 하는 웹 어플리케이션을 제작했다.

논문을 직접 읽고 ([논문 번역](https://doheon.github.io/%EB%85%BC%EB%AC%B8%EB%B2%88%EC%97%AD/pt-Informer-post/)),  코드로 구현한 후([코드구현](https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/time-series/ci-5.informer-post/)) 사용하기 쉽게 모듈화까지 한 후 사용했다.

&nbsp;

웹 프로젝트를 여러 번 진행했었지만 지금까지는 Django하나만 사용해서 제작했었다. 하지만 이번에는 처음으로 Flask로 백엔드를 구성하고 vue.js로 프로트엔드를 구성해서 제작했다.

그 이유는 주식 api가 파이썬 32bit에서만 작동하고, 딥러닝 라이브러리들은 32bit에서 작동하지 않아서 백엔드를 따로 구성해야 했는데 Django 프로젝트를 두 개 제작하는 것보다 Flask로 api만 따로 만들고 vue.js로 화면을 구성하는게 더 나을 것 같아서였다. 해보고 나니까 프론트엔드 프레임워크를 다룰줄 안다면 그냥 이 방법이 더 좋은것 같다는 생각이 들었다.

Flask를 이용해서 필요한 api들을 배포하고, vue.js에서 axios를 이용해서 배포된 api와 통신해서 정보를 가져오는 방식으로 제작했다.

&nbsp;

현재 웹 페이지는 무료 호스팅 서비스를 이용해서 배포는 했지만, Flask api들은 주식 로그인 정보와 GPU가 필요해서 내 컴퓨터에서만 배포가 가능하기 때문에 내 컴퓨터에 배포했고, 내 컴퓨터가 켜져있을 때만 정상적으로 서비스가 이용 가능하다. Flask api는 waitress를 사용해서 배포했다.

어차피 나 혼자 사용할거 같아서 항상 서버를 켜놓지는 않을 것 같다.

[배포링크](http://stocker.dothome.co.kr/)

&nbsp;



## Overview

![main](/assets/images/2021-08-26-dev-StockVue.assets/main.gif)

웹 페이지의 구동과정은 대략 위와 같다.

&nbsp;

원하는 주식을 검색해서 최근 주가와 실시간 주가를 확인할 수 있고, 예측 버튼을 누르면 2주치 미래의 예측 주가를 그래프로 확인할 수 있다.

&nbsp;



## Backend

먼저 두 개의 Flask프로젝트를 이용해서 두 가지 목적의 api를 제작했다.

첫 번째는 신한금융투자의  api를 이용한 주식 정보를 가져오는 api이고, 두 번째는 Informer를 모듈화하여 제작한 시계열 예측 api이다. 

&nbsp;

### Stock Info

금융 관련 api가 모두 그런건지는 모르겠지만, 신한 api는 PyQt5라는 모듈을 기반으로 작동하는데 GUI를 기본으로 사용하도록 되어 있고 사용방법이 생각보다 복잡해서 데이터만을 리턴하도록 만드는 과정이 조금 번잡했다.

조회하는 과정들이 기본적으로 main에서만 실행이 되서 각각의 기능마다 파일을 만들어서 파일을 실행하는 방식으로 구현했다.

3개의 request가 존재한다.

&nbsp;

**/getdata - POST**

- input: 시작점, 끝점, 시간간격, 개수, 주식명

- output: 해당하는 시간과 종목의 주식정보



최근 30일간의 주가정보에 대한 그래프와, 오늘의 주가에 대한 그래프를 그릴 때 사용한다.

&nbsp;



**/getrtdata - POST**

- input: 주식명

- output: 현재 실시간 가격



실시간 주가를 표시해 줄 때 사용한다.

&nbsp;



**/getDict - GET**

- output: 모든 주식코드와 주식명에 대한 정보, 가져온 날짜

 자동완성 기능을 구현할 때 사용한다. 한번 가져오면 local storage에 저장해서 다음부터는 통신을 할 필요가 없게 만들었는데, 가져온 날짜가 달라지면 업데이트를 해주도록 했다.

&nbsp;

### Predict

전에 코드로 구현해 봤던 Informer를 클래스로 만들어서 사용하기 쉽게 모듈화한 후, 데이터를 넣으면 바로 예측을 할수 있도록 했다.

progress bar를 만들고 싶어서 특정 epoch만큼만 학습시키고, 학습된 상태를 dictionary에 저장한 후 다음 request에서 이어서 학습을 진행할 수 있도록 구현했다.

1개의 request가 존재한다.

&nbsp;

**/predict - POST**

- input: 시계열 데이터, 학습의 고유한 id, 마지막 반복인지 여부

- output: 예측된 데이터, 마지막 가격대비 변한 정도

input으로 입력받은 학습의 고유한 id가 dicitonary에 존재하지 않는다면 모델을 생성해서 특정 epoch만큼 학습을 하고, 이미 dictionary에 존재하는 id라면 이전 학습상태에서 이어서 학습을 진행하도록 했다.

마지막 반복인지 여부도 입력받아서 만약 마지막 반복이라면, dictinoary에서 모델을 삭제해서 메모리를 아낄 수 있도록 했다.

&nbsp;





## Frontend

vue.js를 이용하여 화면을 구성했으며, vue router로 종목검색, 즐겨찾기 두가지의 메뉴를 구현했다.

구현한 기능의 일부는 아래와 같다.

&nbsp;

### 검색어 자동 완성

![image-20210825225543526](/assets/images/2021-08-26-dev-StockVue.assets/image-20210825225543526.png)

검색 기능에서 자동완성은 필수라고 생각해서 구현했는데 생각보다 시간이 오래 걸렸던 것 같다.

한글의 자음 모음을 분해해주는 모듈을 이용해서 검색어를 분해한 후 모든 주식 종목 중 검색어가 포함되는 주식을 뽑아낸 후 겹치는 위치가 빠른 것을 기준으로 정렬해서 15개만 출력하도록 했다.

모든 주식 종목 리스트는 api로 얻어오며 한번만 얻어오면 결과를 local storage에 저장해서 다음부터는 더 빠르게 작동할 수 있도록 했다. 주식 종목 리스트는 시간이 지나면 바뀔 수도 있으므로, 마지막으로 가져온 날짜가 오늘과 다르다면 다시 api로 얻어오게 해서 하루에 한번은 가져오도록 했다.

선택하는 커서 설정, 키보드 화살표로 커서이동, 마우스 hover로 커서이동, 엔터나 클릭시 바로 검색 등 간단해 보이는 기능이지만 구현해야할 부분이 많았다.

&nbsp;

### 검색 화면

![image-20210825225609887](/assets/images/2021-08-26-dev-StockVue.assets/image-20210825225609887.png)

검색 결과는 종목멱과 현재 가격을 알려주는 부분, 근 30일간의 일간 주가를 알려주는 부분, 오늘의 시간별 실시간 주가를 알려주는 부분으로 이루어져 있으며 각각 component를 나눠서 종목명만 있으면 표시되도록 구현했다.

그래프는 chart.js를 사용해서 구현했다.

&nbsp;



### 예측 결과

![image-20210825230237698](/assets/images/2021-08-26-dev-StockVue.assets/image-20210825230237698.png)

예측 버튼을 누르면 api와 통신 후 예측 결과를 받아와서 그래프로 보여준다.

예측하는 진행도를 원형 progress bar로 표시해 준다. 

예측이 끝나면 가장 최근 주가 대비 예측한 주가들의 평균이 오르는지 내리는지를 알려주도록 했다.

&nbsp;





![image-20210826101657569](/assets/images/2021-08-26-dev-StockVue.assets/image-20210826101657569.png)

실시간 주가를 나타내는 아래 그래프에서 예측 버튼을 누르면 현재 시간 부터 오늘의 장 마감까지의 주가를 예측해서 보여준다. 그래서 장 중에만 사용이 가능하다.

&nbsp;



### 즐겨찾기

![image-20210825231651128](/assets/images/2021-08-26-dev-StockVue.assets/image-20210825231651128.png)



![image-20210825231527120](/assets/images/2021-08-26-dev-StockVue.assets/image-20210825231527120.png)



검색결과 옆에 즐겨찾기 버튼이 있는데 이 버튼을 눌러 놓으면 즐겨찾기 메뉴에서 실시간 주가를 한눈에 확인할 수 있도록 했다. 

즐겨찾기에서 클릭하면 바로 검색이 되도록 했다.

즐겨찾기 리스트를 local storage에 저장하는 방식으로 구현했다.

&nbsp;



### 모바일 최적화

![mobile](/assets/images/2021-08-26-dev-StockVue.assets/mobile.gif)

주식을 본다면 보통 핸드폰으로 볼것 같아서 모바일 화면에서도 잘 볼 수 있도록 반응형 웹을 구현했다.

&nbsp;



## Conclusion

지금까지 웹 프로젝트를 할 때는 연습한다는 느낌으로 했었는데, 이번에는 진짜 사람들에게 유용한 서비스를 만들어 본다는 마음가짐으로 진행했던 것 같다. 물론 진짜 서비스할만큼의 수준은 아니지만 어느정도 만족할만큼의 결과는 나온것 같다.

느낀점은 프론트엔드 프레임워크를 사용하는게 훨씬 더 다양한 기능을 구현할수 있다는 것과,  분업을 한다고 가정한다면 Django만 사용해서 제작했을 때보다 더 효율적인 부분이 많다는 생각이 들었다.

내가 주식을 했다면 어떤 기능이 필요한지 더 잘 알 수 있었을 텐데 주식을 잘 몰라서 어떤 기능이 더 있어야 하는지 잘 몰라서 조금 아쉬웠다.

주식 api를 사용하려고 증권계좌를 만들었는데 나중에 매수, 매도 기능까지 있는 버전을 만들어서 직접 사용해 볼 생각이다.



