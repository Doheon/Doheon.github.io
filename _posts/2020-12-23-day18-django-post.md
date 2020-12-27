---
title: "[인공지능 데브코스] 4주차 day3 - django"
toc: true
toc_sticky: true
date: 2020-12-23
categories: TIL
---



## 12월 23일 수   
파이썬 기반 웹 프레임 워크 django를 공부했다. flask와 비슷한 점이 있었지만 이미 구현되어 있는 부분이 많아 공부할게 더 많은 것 같다. 하지만 직접 구현해야 할게 적어서 더 좋은 것 같다.  


##  django 시작하기

python 기반 웹 프레임워크  
pinterest, instagram  

flask와 비교  
flask: “마이크로”  
django  

django-admin startproject projname : 프로젝트 생성  
python manage.py runserver : 서버가동  
<p>&nbsp;</p>  

## django의 구성요소

django Project and App  
한 project는 여러 App으로 구성되어 있다.  

django-admin startapp appname : app생성, 해당 프로젝트안의 경로에 있어야 한다  

__init__py : 모듈로 인식  
admin.py : admin page제공  
apps.py : app에 대한 설정  
models.py : 데이터베이스의 스키마를 클래스 형태로 저장  
test.py : 테스트케이스 설명  
views.py: view관리  


**django의 MVT Pattern**  
View, Model, Template  

![Alt Text](/assets/images/django/django1.png)  
<p>&nbsp;</p>  


## View 로 Request Handling하기

app의 views.py 에서 함수를 정의한다.  

project의 urls.py에서 해당 함수를 import한다.  

settings.py에 INSTALLED_APPS에 해당 app을  추가해준다.  
<p>&nbsp;</p>  

**과정**  
사용자가 127.0.0.1에 요청을 보냄  
urls.py에서 패턴이 path에 있는지 확인  
views에서 해당 함수를 호출  
<p>&nbsp;</p>  


**admin**  
django는 기본적으로 admin 페이지를 제공한다.  
로그인을 진행하면 앱에대한 관리가 가능하다.  
<p>&nbsp;</p>  


python manage.py migrate  
python manage.py createsuperuserpy  
admin 계정 생성  

해당 계정으로 로그인후 사이트를 관리할 수 있음  
<p>&nbsp;</p>  


## Template으로 보여줄 화면 구성하기

view-> template language -> html, css, javascript  

render(request,  ‘.html’, {} )  

html파일들을 넣어 놓을 파일을 만들어 놓고  
settings.py의 TEMPLATES의 DIRS를 업데이트 해야 한다.  
BASE_DIR에 이 웹프로젝트의 경로가 저장되어있는데 이를 이용한다.  
<p>&nbsp;</p>  

**render의 세번째인자**  
dic형태로 값을 저장한 후  
html파일에서 {{ key }} 와같이 중괄호 두번안에 dic의 key값을 넣어주면 dic의 value가 나온다.  
외부의 값을 이용해야 할때 사용  
<p>&nbsp;</p>  

html만 사용하면 내용을 정해진 것만 할수 있지만  
template언어를 사용하면 직접 데이터를 넣어 줄 수 있다.  
=> 동적인 웹사이트를 만들 수 있다.  

<p>&nbsp;</p>  

**template filter**  
{{ key | filter }}와 같이 사용하며 value에 filter를 적용한 값이 나온다.   
length : 길이 출력  
upper: 대문자로 출력  

<p>&nbsp;</p>  

**template tag**  
중괄호를 이용한 형식으로 사용하며 html에 로직을 도입할 수 있다  
