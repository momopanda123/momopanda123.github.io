---
layout: default
title: 02. Gabor Filter
subtitle: Image Processing
---
-----

[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/02_GaborFilter)

-----

# Gabor Filter
- ����ΰ������а� ���п�
    2022254026
        ��ȫ��


---

# **Gabor Filter��?**

Gabor Filter�� ����ó������ Bio-inspired��� Ű���尡 ������ �������ʰ� �����Ѵ�.

�ܰ����� �����ϴ� ����� �ϴ� ���ͷ�, ����� �ð�ü�谡 �����ϴ� �Ͱ� ����ϴٴ� ������ �θ� ���ǰ� �ִ�.

Gabor Fiter�� ������ ���ؼ� ���� �Լ��� ��ⷹ�̼� �� Gaussian Filter��� ������ �� �ִ�.

�Ķ���͸� �����Կ� ���� Edge�� ũ�⳪ ���⼺�� �ٲ� �� �����Ƿ� Bio-inspired ����ó�� �˰��򿡼� Ư¡�� ���� �˰������� �ٽ����� ������ �ϰ� �ִ�.

2D Gabor Filter�� ������ �Ʒ��� ����.


![gaborfilter](./images/gaborfilter.png)
![gaborfilter](./images/gaborfilter2.png)
![gaborfilter](./images/gaborfilter3.png)


```cpp

cv::Mat cv::getGaborKernel(cv::Size ksize, double sigma, double theta, double lambd, double gamma, double psi = CV_PI*0.5, int ktype = CV_64F)


```

cv::getGaborKernel �Լ��� OpenCV���� ��������(Gabor filter)�� �����ϴ� �� ���ȴ�.

�������ʹ� �̹��� ó���� ��ǻ�� �������� Ư�� ���⼺�� ���ļ��� Ư¡�� �����ϴ� �� ���Ǵ� ���� �����̴�. 


### **Parameters**

* ksize: Ŀ���� ũ���, cv::Size Ÿ��. Ŀ���� �ʺ�� ���̸� ����.

* sigma: ����þ� �Լ��� ǥ�� ����. ���� Ŀ������ Ŀ���� ũ�Ⱑ Ŀ����, ������ ������ ��������.

* theta: ������ ������ ���� ������ �����Ѵ�. 0�� ���� ����, CV_PI/2�� ���� ������ �ǹ��Ѵ�.

* lambd: ���������� ����. �̹����� Ư�� ���ϰ� �󸶳� �� ��ġ������ �����Ѵ�.

* gamma: ������Ⱦ��(Spatial aspect ratio)��, Ÿ���� ���� �Լ��� Ÿ������ �����Ѵ�. (gamma=1�� ����, gamma<1�� Ÿ����)

* psi: ������(Phase offset)��, ���� ������ ������ �����Ѵ�. (�Ϲ������� CV_PI*0.5�� �⺻������ ���)

* ktype: Ŀ�� Ÿ��. (���� CV_64F (64-��Ʈ �ε��Ҽ���)�� ���)




---

### ���� �ڵ�[��]()

<details>
<summary>Import</summary>
<div markdown="1">
  
```python



```

</div>
</details>


---

### ����[��]()

- Gabor Filter - Google
- ChatGPT
- [Blog](https://thinkpiece.tistory.com/304)
