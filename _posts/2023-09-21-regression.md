---
layout: default
title: 05. Regression
subtitle: 어프렌티스 프로젝트 과목
---
-----

[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/05_Regression)

-----

# 05. Regression
- 산업인공지능학과 대학원
    2022254026
        김홍열

---


# *Regression*
### **Linear Regression, Logistic Regression**

# 선형 회귀 (Linear Regression)

선형 회귀는 연속적인 값을 예측하기 위한 회귀 알고리즘입니다. 주로 종속 변수와 독립 변수 간의 선형 관계를 모델링하는 데 사용됩니다.

### 수식
​ \(y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon\)
  - \(y\): 예측값
  - \(x_1, x_2, ..., x_n\): 독립 변수들
  - \(\beta_0, \beta_1, ..., \beta_n\): 회귀 계수
  - \(\epsilon\): 오차 항
  
### 목적
오차 항 ϵ의 제곱합을 최소화하는 회귀 계수를 찾는 것입니다.

### 사용 사례
주택 가격 예측, 연봉 예측, 판매량 예측 등 연속적인 값을 가지는 대상을 예측할 때 사용됩니다.

---

### 예제 코드[¶]()

<details>
<summary>Code View</summary>
<div markdown="1">
  
```python

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# 예제 데이터 생성
X = np.random.rand(100, 1) * 10  # 100개의 랜덤 데이터
y = 2.5 * X + 5 + np.random.randn(100, 1) * 2  # y = 2.5x + 5 + 잡음

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


```

</div>
</details>

<details>
<summary>diabetes 데이터셋 활용</summary>
<div markdown="1">


### 개요
    - 위스콘신 대학의 유방암 진단 데이터셋으로, 유방암 종양의 임상 정보를 기반으로 악성(malignant) 또는 양성(benign)으로 분류하는 문제에 사용됩니다.

### 특징
    - 30개의 특징 변수가 있으며, 이는 종양의 다양한 특성(크기, 반경, 질감 등)을 나타냅니다.

### 목표 변수
    - 종양이 악성인지(1) 양성인지(0)를 나타내는 이진 값입니다.

### 용도
    - 이 데이터셋은 주로 분류 문제에 사용됩니다.

### 속성
    - age: 나이
    - sex: 성별
    - bmi: 체질량지수
    - bp: 평균 혈압
    - s1 ~ s6: 6개의 혈청 측정값


<details>
<summary>Code View</summary>
<div markdown="1">


```python

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 로드
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  # BMI feature만 사용
y = diabetes.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

```

</div>
</details>
</div>
</details>

---

# 로지스틱 회귀 (Logistic Regression)
로지스틱 회귀는 이름에 '회귀'가 들어가지만, 분류 문제에 사용되는 알고리즘입니다. 주로 이진 분류 문제에 사용되며, 확률을 출력으로 가집니다.

### 수식
  - \(P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n)}}\)
  - \(P(Y=1)\): 클래스 1에 속할 확률
  - \(x_1, x_2, ..., x_n\): 독립 변수들
  - \(\beta_0, \beta_1, ..., \beta_n\): 회귀 계수

### 목적
  - 로그 오즈를 최대화하는 것입니다. 로그 오즈는 실제 값과 예측 확률 간의 로그 비율로 계산됩니다.

### 사용 사례
  - 스팸 메일 분류, 환자의 질병 발병 여부 예측, 고객 이탈 여부 예측 등 이진 분류 문제에 주로 사용됩니다. 다중 클래스 분류에도 확장하여 사용할 수 있습니다.
---


<details>
<summary>Code View</summary>
<div markdown="1">


```python


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 예제 데이터 생성
X = np.random.rand(100, 1) * 10  # 100개의 랜덤 데이터
y = (X > 5).astype(int).ravel()  # X가 5보다 크면 1, 아니면 0

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


```

</div>
</details>


<details>
<summary>breast_cancer 데이터셋 활용</summary>
<div markdown="1">

### 개요
  - 위스콘신 대학의 유방암 진단 데이터셋으로, 유방암 종양의 임상 정보를 기반으로 악성(malignant) 또는 양성(benign)으로 분류하는 문제에 사용됩니다.

### 특징
  - 30개의 특징 변수가 있으며, 이는 종양의 다양한 특성(크기, 반경, 질감 등)을 나타냅니다.

### 목표 변수
  - 종양이 악성인지(1) 양성인지(0)를 나타내는 이진 값입니다.

### 용도
  - 이 데이터셋은 주로 분류 문제에 사용됩니다.
---


<details>
<summary>Code View</summary>


```python

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 로드
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=10000)  # max_iter를 증가시켜 수렴을 도움
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


```

<div markdown="1">
</div>
</details>
</div>
</details>

---

### 참고[¶]()

- 어프렌티스 프로젝트 과목, 김재영 교수
- ChatGPT
