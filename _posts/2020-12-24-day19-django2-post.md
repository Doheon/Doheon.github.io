---
title: "[인공지능 데브코스] 4주차 day4 - django2"
toc: true
toc_sticky: true
date: 2020-12-24
categories: TIL
---

## 12월 23일 목   

django로 동적 웹페이지를 만들었다.  


## MVT

**model**  
model은 database를 관리  

**Database**
데이터를 저장하는 시스템  
구조화한다는 점이 특징  
단순히 정보를 저장하는 창고가 아니라  
사용하기 쉽게 정렬해 놓는다.  

**Relational DB**  
가장 많이 사용하는 DB  
테이블 형태로 데이터를 보관  

**SQL**  
DB에서 정보를 가져오는 interface역할을 하는 프로그래밍 언어  

django는 sql을 사용안해도 데이터베이스에 접근이 가능  
=> ORM  

django은 orm이 내장되어 있다.  

모델은 models.py에서 관리  
class로 모델을 관리  

**model.py**  
class <model_name> (models.Model):  
Field 1  
Field 2  
field의 타입을 지정  
문자열: CharField  
숫자: IntegerField, SmallIntegerField,   
논리형: BooleanField  
시간/날짜: DateTimeField  

각 field의 옵션  
default: 기본적으로 어떤값이 들어있는지  
null: 값이 비어있을 수 있는지  
<p>&nbsp;</p>  

**admin.py**  
모델이 있을 때 모델을 자연스럽게 관리해 줄 수 있다.  
from .models import model_name  
admin.site.register(model_name)  
를 추가하면 admin페이지에서 model을 관리할 수 있다.  
(기존에 있었던 Groups, Users도 모델이다)  
<p>&nbsp;</p>  


python manage.py makemigrations homepage  
git add와 비슷한 것으로 migrate할 항목을 업데이트 한다.  

python manage.py migrate  
를 해줘야 클래스 형태의 모델이 연동되어 admin에서 사용할 수 있다.  

git add -> git commit과 비슷한 작업  

이제 admin사이트에서 직접 데이터를 추가해 줄 수 있다.  

def __str__(self):  
return self.name  
이 객체를 출력하는 과정에서 어떤 문자열을 보여줄지 결정하는 함수  


1. models.py에서 모델 생성
2. views.py에서 모델 import후 html로 데이터를 옮겨주는 함수 작성
3. html파일 생성후 전달받은 데이터 출력

<p>&nbsp;</p>  

## 템플릿에서 모델의 정보를 갱신하기

forms.py 생성  

from django import forms  
from .models import Coffee #모델 호출  

class CoffeeForm(forms.ModelForm): #어떤 모델에 대해서 입력칸을 만들어 주는 객체   
class Meta:  
model = Coffee  
fields = (’name’, ‘price’, ‘is_ice’)  

이 클래스를 views.py에 전달  
veiws.py에서 template로 전달  

template에서 form을 이용해서 정보를 얻음  

coffee_form.as_p  형태를 바꿔줌  