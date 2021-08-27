---
title: "[개발] Time Series Imputation - Django"
toc: true
toc_sticky: true
date: 2021-08-26
categories: 개발 web Django
---

**시계열 결측치 대체 웹 프로젝트**

&nbsp;

이전에 time series에서의 결측치를 대체하는 두 논문을 읽고 교통 데이터에서 결측치를 대체하는 코드를 구현했었다.  ([BRITS](https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/time-series/ci-1.brits-post/), [NAOMI](https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/time-series/ci-2.naomi-post/))

&nbsp;

구현한 코드를 모듈화해서 사용하기 쉽게 만든 후, 한번의 epoch가 진행될 때마다 결측치가 대체된 상태를 직접 확인하면서 훈련을 진행할 수 있도록 UI와 시각화를 구현한 Django 웹 페이지를 제작해 보았다.

다른 프론트엔드 프레임워크 없이 Django만을 사용해서 제작했다.

&nbsp;

배포는 되어 있지 않고, 코드는 아래에서 확인할 수 있다.

[github](https://github.com/Doheon/ImputationDjango)

&nbsp;



![demo2](/assets/images/2021-08-26-dev-ImputationDjango.assets/demo2.gif)

최종적인 웹페이지의 실행화면은 대략 위와 같다.

&nbsp;



## Backend

Django의 views.py에는 프론트의 request를 받는 함수들과 util 함수들이 존재한다.

&nbsp;

### util

```python
modelDic = {}

#해당하는 uid의 모델을 returngksek.
def getModel(uid, method=None, dataframe = None, window_size = None):
    #modelDic에 이미 모델이 있다면 바로 리턴해 준다.
    if uid in modelDic:
        return modelDic[uid]
	
    #그게 아니라면 처음 학습을 시작하는 것이므로 모델을 생성해준다.
    #NAOMIimputation과 BRITSimputation은 사용한 두 방법을 모듈화를 해서 만든 클래스로, 다른 파일에서 선언한 후 import했다.
    if method=="NAOMI":
        modelDic[uid] = (NAMOIimputation(dataframe = dataframe, window_size=window_size), "NAOMI")
    elif method=="BRITS":
        modelDic[uid] = (BRITSimputation(dataframe = dataframe), "BRITS")
    else:
        find = autoFind(dataframe)
        if find:
            modelDic[uid] = (NAMOIimputation(dataframe = dataframe, window_size=len(dataframe)//10), "NAOMI")
        else:
            modelDic[uid] = (BRITSimputation(dataframe = dataframe), "BRITS")
    print(modelDic)
    return modelDic[uid]

#method를 auto로 설정할 경우 결측치의 구간에 따라 자동으로 방법을 선택해 준다.
#NAOMI가 긴 결측치에 좋다는 성질을 반영하기 위해 만들었다.
def autoFind(dataframe):
    maxnull = 0
    cur = 0
    for i in range(len(dataframe)):
        if dataframe["value"].iloc[i] == "NaN" or pd.isnull(dataframe["value"].iloc[i]):
            cur += 1
            maxnull = max(maxnull, cur)
        else:
            cur = 0
    return maxnull>10
```

학습을 한번에 진행하는게 아니라 한번의 epoch씩 진행하므로 모델의 학습 상태가 서버에 유지되고 있어야 이어서 학습을 진행할 수가 있다. 

따라서 학습을 할 모델들이 들어있는 models라는 dictionary를 전역변수로 선언한 후, 하나의 학습 과정마다 가지고 있는 고유한 key를 이용해서 모델을 불러오는 방식으로 이어서 학습을 진행할 수 있도록 했다.

getModel함수는 이러한 과정을 구현한 함수고, autoFind는 방법으로 auto로 설정했을 때 결측치의 길이를 기준으로 방법을 자동으로 설정해 주는 함수이다.

&nbsp;



### index(request)

```python
def index(request):
    return render(request, 'index.html')
```

이 사이트는 index.html하나의 html파일로만 구성되어 있다.

화면의 업데이트는  jquery의 ajax를 이용하여 새로고침 없이 받아오도록 했다.

&nbsp;



### imputation(request)

```python
def imputation(request):
    if len(request.FILES) == 0:
        return
    file = request.FILES["getCSV"]
    window_size = int(request.POST["windowsize"])
    method = request.POST["method"]
    uid = request.POST["uid"]

    df = pd.read_csv(file)
    df = df.fillna("NaN")
	
    #모델을 불러온다.
    model, method = getModel(uid, method, df, window_size)
    
    if method=="NAOMI":
        #한번만 학습을 시킨다.
        model.imputation(1)
        result_df = model.df
        result = model.scaler.inverse_transform(result_df["value"].values.reshape(-1,1)).squeeze()
        result = result.tolist()
        label = result_df["time"].values.tolist()
        dic = {"label":label, "value": result}
        return HttpResponse(json.dumps(dic), content_type = "application/json")
    elif method=="BRITS":
        #한번만 학습을 시킨다.
        result = model.imputation(1)
        result_df = model.df 
        result = result.tolist()
        label = result_df["time"].values.tolist()
        dic = {"label":label, "value": result}
        return HttpResponse(json.dumps(dic), content_type = "application/json")
```

imputation은 처음 학습이 진행될 때 실행되는 함수로, 프론트에서 받아온 각종 정보를 이용하여 모델을 선언하는 함수이다.

model.imputaion(1) 으로 한 번의 epoch만 학습을 진행한 후의 결과를 json형태로 프론트로 전달해 준다.

&nbsp;



### imputeProcess(request)

```python
def imputeProcess(request):
    uid = request.POST["uid"]
    model, method = getModel(uid)
    if method=="NAOMI":
        model.optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.model.parameters()), lr=7e-4)
        model.run_epoch(True, model.model, model.train_data, 10, model.optimizer, batch_size = 64, teacher_forcing=True)

        result = model.predict_result()
        result = model.scaler.inverse_transform(result.reshape(-1,1)).squeeze()
        result = result.tolist()
        label = model.df["time"].values.tolist()
        dic = {"label":label, "value": result}
        return HttpResponse(json.dumps(dic), content_type = "application/json")
    elif method=="BRITS":
        model.run_one_epoch()
        result = model.predict_result()
        result = result.tolist()
        label = model.df["time"].values.tolist()
        dic = {"label":label, "value": result}
        return HttpResponse(json.dumps(dic), content_type = "application/json")
```

imputeProcess는 두 번째 epoch부터 실행되는 함수이다.

학습의 고유한 키인 uid를 이용해서 이전에 학습하던 모델을 가져와서 1번의 epoch로 학습을 추가로 시킨 후 결과를 json형태로 프론트에 전달해 준다.

&nbsp;



### visualize(request)

```python
def visualize(request):
    if len(request.FILES) == 0:
        return
    file = request.FILES["getCSV"]
    df = pd.read_csv(file)
    df = df.fillna("NaN")
    result = df["value"].values.tolist()
    label = df["time"].values.tolist()
    dic = {"label":label, "value": result}
    return HttpResponse(json.dumps(dic), content_type = "application/json")
```

프론트에서 csv파일을 받아와서 데이터를 가져온 후 json형태로 다시 전달해 주는 함수다.

&nbsp;



## Frontend

따로 템플릿을 다운받거나 하지 않고 손수 bootstrap을 이용해서 화면을 구성했다.

훈련방법, epoch, csv파일과 같은 입력 정보들을 서버로 전달해준 후  jquery의 ajax를 이용해서 데이터를 받아오고 chart.js를 이용해서 실시간으로 시각화 시켜 줄 수 있도록 했다.

 java script를 이용하여 훈련시작, 훈련중지, 추가학습과 같은 기능들을 구현해서 실시간으로 결측치가 대체된 결과를 확인하면서 만족할만한 결과가 나올 때까지 학습을 조절 할수 있도록 했다.

이러한 방법은 epoch 한번 진행할 때마다 서버와 통신을 하기 때문에 한번에 모든 epoch를 진행하는 것보다 속도는 느리겠지만, **실시간으로 결과를 확인**한다는 점 때문에 학습 epoch를 조절하는 측면에서 시간을 훨씬 더 단축시킬 수 있을 것이라고 생각한다.



