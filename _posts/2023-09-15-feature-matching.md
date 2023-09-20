---
layout: default
title: 07. Feature Matching
subtitle: 지능자동화실제 과목
---
-----

[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/07_Feature_matching)

-----

# 07. Feature Matching
- 산업인공지능학과 대학원
    2022254026
        김홍열


# Feature Matching이란
* 컴퓨터 비전 및 이미지 처리 분야에서 두 이미지 간의 유사한 특징을 찾는 과정
* 두 이미지 간의 관계 (예: 동질성 행렬, 기본 행렬)를 추정하거나, 특정 객체의 위치와 자세를 찾아내는 데 사용
* 객체 인식, 이미지 스티칭, 3D 재구성 등 다양한 응용에서 활용


# Featrue Matching 과정

1. **Feature Detection (특징 검출)**
    * 각 이미지에서 특징점 (코너, 블롭 등)을 검출
    * ex) SIFT, SURF, ORB, FAST, Harris Corner Detector 등

2. **Feature Description (특징 기술)**
    * 각 특징점 주변의 정보를 기술자(descriptor)로 변환
    * 기술자는 해당 특징점의 고유한 정보를 포함하며, 다른 이미지의 특징점과 비교할 때 사용
    * ex) SIFT, SURF, ORB 등

3. **Feature Matching (특징 매칭)**
    * 한 이미지의 특징 기술자와 다른 이미지의 특징 기술자를 비교하여 유사한 특징들을 검출
    * ex) Brute-Force 매칭, FLANN 기반 매칭

4. **Outlier Removal (이상치 제거)**
    * 매칭 과정에서 잘못된 매칭을 이상치(outliers)라고 하며, 이를 제거하는 과정
    * ex) RANSAC, LMEDS 등


---

### 예제 코드[¶]()

<details>
<summary>Harris Corner</summary>

<div markdown="1">
  
```c++

void corner_fast()
{
    Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

    if (src.empty())
    {
        cerr << "Image load failed !" << endl;
        return;
    }

    vector<KeyPoint> keypoints;
    FAST(src, keypoints, 60, true);

    Mat dst;
    cvtColor(src, dst, COLOR_GRAY2BGR);

    for (KeyPoint kp : keypoints)
    {
        Point pt(cvRound(kp.pt.x), cvRound(kp.pt.y));
        circle(dst, pt, 5, Scalar(0, 0, 255), 2);
    }

    imshow("src", src);
    imshow("dst", dst);
    imwrite("dst1.png", dst);

    waitKey(0);
    destroyAllWindows();
}

```

![Origin](/assets/img/building.jpg)
![Result](/assets/img/dst1.png)

</div>
</details>


<details>
<summary>ORB</summary>

<div markdown="1">
  
```c++

void detect_keypoints()
{
    Mat src = imread("box_in_scene.png", IMREAD_GRAYSCALE);

    if (src.empty())
    {
        cerr << "Image load failed !" << endl;
        return;
    }

    Ptr<Feature2D> feature = ORB::create();

    vector<KeyPoint> keypoints;
    feature->detect(src, keypoints);

    Mat desc;
    feature->compute(src, keypoints, desc);

    cout << "keypoints.size(): " << keypoints.size() << endl;
    cout << "desc.size(): " << desc.size() << endl;

    Mat dst;
    drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow("src", src);
    imshow("dst", dst);
    imwrite("dst2.png", dst);

    waitKey();
    destroyAllWindows();
}

```

![Origin](/assets/img/box_in_scene.png)
![Result](/assets/img/dst2.png)

</div>
</details>


<details>
<summary>ORB Matching</summary>

<div markdown="1">
  
```c++

void keypoint_matching()
{
    Mat src1 = imread("box.png", IMREAD_GRAYSCALE);
    Mat src2 = imread("box_in_scene.png", IMREAD_GRAYSCALE);

    if (src1.empty() || src2.empty())
    {
        cerr << "Image load failed !" << endl;
        return;
    }

    Ptr<Feature2D> feature = ORB::create();

    vector<KeyPoint> keypoints1, keypoints2;
    Mat desc1, desc2;
    feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
    feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

    vector<DMatch> matches;
    matcher->match(desc1, desc2, matches);

    Mat dst;
    drawMatches(src1, keypoints1, src2, keypoints2, matches, dst);

    imshow("dst", dst);
    imwrite("dst3.png", dst);

    waitKey();
    destroyAllWindows();
}

```

![Origin1](/assets/img/box.png)
![Origin2](/assets/img/box_in_scene.png)
![Result](/assets/img/dst3.png)

</div>
</details>


### 참고[¶]()

- 지능자동화실제 과목, 박태형 교수
- ChatGPT
