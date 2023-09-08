---
layout: default
title: 01. SCREENSHOT OCR
subtitle: 모니터 화면을 캡쳐해서 원하는 영역을 ROI 설정 후, Tesseract로 OCR 변환하여 문자 출력
---

# 01. SCREENSHOT OCR
### [Tesseract](https://github.com/tesseract-ocr/tesseract)
- 이 프로그램은 OCR 엔진인 libtesseract와 커맨드 라인 프로그램인 tesseract를 포함하고 있습니다.
Tesseract 4는 신경망 (LSTM) 기반의 OCR 엔진으로 traineddata 폴더 안에 구성요소들이 있습니다.

- OpenCVSharp4를 이용하여 모니터 화면의 캡쳐 영역을 ROI하여 Tesseract와 연동하였습니다.

- 현재 English 모델만 구성하였고, 필요에 따라서 다른 언어의 문자 인식모델들을 추가할 수 있습니다.

-----


[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/01_ScreenShotOCR)

-----


### 사용법

![How_to_use](assets/img/How_to_use.gif)
