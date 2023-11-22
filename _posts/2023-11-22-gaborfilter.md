---
layout: default
title: 02. Gabor Filter
subtitle: Image Processing
---
-----

[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/02_GaborFilter)

-----

# Gabor Filter
- 산업인공지능학과 대학원
    2022254026
        김홍열


---

# **Gabor Filter란?**

Gabor Filter는 영상처리에서 Bio-inspired라는 키워드가 있으면 빠지지않고 등장한다.

외곽선을 검출하는 기능을 하는 필터로, 사람의 시각체계가 반응하는 것과 비슷하다는 이유로 널리 사용되고 있다.

Gabor Fiter는 간단히 말해서 사인 함수로 모듈레이션 된 Gaussian Filter라고 생각할 수 있다.

파라미터를 조절함에 따라 Edge의 크기나 방향성을 바꿀 수 있으므로 Bio-inspired 영상처리 알고리즘에서 특징점 추출 알고리즘으로 핵심적인 역할을 하고 있다.

2D Gabor Filter의 수식은 아래와 같다.


![gaborfilter](/assets/img/gabor/gaborfilter.png)

![gaborfilter](/assets/img/gabor/gaborfilter2.png)

![gaborfilter](/assets/img/gabor/gaborfilter3.png)


```cpp

cv::Mat cv::getGaborKernel(cv::Size ksize, double sigma, double theta, double lambd, double gamma, double psi = CV_PI*0.5, int ktype = CV_64F)


```

cv::getGaborKernel 함수는 OpenCV에서 가버필터(Gabor filter)를 생성하는 데 사용된다.

가버필터는 이미지 처리와 컴퓨터 비전에서 특정 방향성과 주파수의 특징을 강조하는 데 사용되는 선형 필터이다. 


### **Parameters**

* ksize: 커널의 크기로, cv::Size 타입. 커널의 너비와 높이를 지정.

* sigma: 가우시안 함수의 표준 편차. 값이 커질수록 커널의 크기가 커지고, 필터의 감도가 낮아진다.

* theta: 필터의 방향을 라디안 단위로 지정한다. 0은 수평 방향, CV_PI/2는 수직 방향을 의미한다.

* lambd: 가버필터의 파장. 이미지의 특정 패턴과 얼마나 잘 매치할지를 결정한다.

* gamma: 공간종횡비(Spatial aspect ratio)로, 타원형 가버 함수의 타원성을 결정한다. (gamma=1은 원형, gamma<1은 타원형)

* psi: 위상변위(Phase offset)로, 가버 필터의 위상을 조절한다. (일반적으로 CV_PI*0.5가 기본값으로 사용)

* ktype: 커널 타입. (보통 CV_64F (64-비트 부동소수점)를 사용)




---

### 예제 코드[¶]()

<details>
<summary>Import</summary>
<div markdown="1">
  
```python



```

</div>
</details>


---

### 참고[¶]()

- Gabor Filter - Google
- ChatGPT
- [Blog](https://thinkpiece.tistory.com/304)
