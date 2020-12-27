---
title: "[인공지능 데브코스] 4주차 day1 - flask"
toc: true
toc_sticky: true
date: 2020-12-21
categories: TIL
---

## 12월 21일 월   

파이썬 기반 웹 프레임워크인 flask에 대해 공부했다. flask를 이용해서 REST API를 구현해 봤고 sqlalchemy를 이용하여 db연동도 구현하였다.   


## 인터넷과 웹  

**인터넷(internet)**  
전 세계 컴퓨터를 하나로 합치는 거대한 통신망  
<p>&nbsp;</p>  

**웹(Web)**  
인터넷에 연결된 사용자들이 정보를 공유할 수 있는 공간  

웹은 클라이언트와 서버 사이의 소통이다  
(두 컴퓨터간의 상호 작용)  

1. client가 server에 정보를 요청한다.  
2. server는 이 요청받은 정보에 대한 처리를 진행한다.   
3. server가 client에게 요청에 대해 응답한다.   

http.request  
http.response  
<p>&nbsp;</p>  


## Flask with Rest API  

**API**  
프로그램들이 서로 상호작용하는 것을 도와주는 매개체  
<p>&nbsp;</p>  

**Representational State Transfer**  
웹 서버가 요청을 응답하는 방법론 중 하나  

데이터가 아닌, 자원의 관점으로 접근  

REST API  
HTTP URI통해 자원을 명시하고  
HTTP Method를 통해 해당 자원에 대한 CRUD를 진행  

REST API의 Stateless  
client의 context를 서버에서 유지하지 않는다   
