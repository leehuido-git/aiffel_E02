# aiffel_E02
Exploration_2, Using various datasets(Iris, wine, breast cancer) using scikit-learn.    
AIFFEL교육과정 중 두번째 Exploration으로 sckit-learn모듈에 있는 Decision Tree, Random Forest, Support Vector Machine (SVM), Stochastic Gradient Descent Classifier (SGDClassifier), Logistic Regression을 통해 어떤 알고리즘이 각 데이터셋(digits, wine, breast cancer)에 가장 좋은 성능을 내는지 확인.

## 개요
- 이미지들 변환    
    본 시스템은 크게 4가지 단계로 이루어져 있습니다.
    1. data pre-processing    
        - sckit-learn을 통해 각 데이터셋 다운로드합니다.    
        - 데이터셋을 train, test를 분류합니다.    
    2. 학습    
        - Decision Tree, Random Forest, Support Vector Machine (SVM), Stochastic Gradient Descent Classifier (SGDClassifier), Logistic Regression를 trainset을 통해 학습합니다.
    3. 예측   
        - 위 모델들을 testset을 통해 예측합니다.
    4. 각 모델 평가    
        - sckit-learn의 classification_report를 통해 각 모델의 가중 평균을 낸 f1-score를 통해 데이터셋에 대한 성능을 비교하고 최종 모델을 도출합니다.

## Installation
파이썬 개발 환경으로 최신 버전의 Anaconda를 설치하세요. (Python3 버전용)
* numpy
* sckit-learn
* matplotlib

```
$ pip install -r requirements.txt
```

------------
## Directory
필수 디렉토리는 다음과 같습니다
```
.
├── E02_01_digits/
│   ├── [E-02]digits.ipynb
│   └── [E-02]digits.py
├── E02_02_wine/
│   ├── [E-02]wine.ipynb
│   └── [E-02]wine.py
└── E02_03_breast_cancer/
    ├── [E-02]breast_cancer.ipynb
    └── [E-02]breast_cancer.py

```


## 차별점, 문제점
1. [E-02]digits에서 실제 라벨과 이미지가 맞는지 확인했습니다.
2. weighted avg f1 socre를 통해 각 모델의 성능을 평가해 가장 좋은 알고리즘을 찾았습니다.
3. f1-score에 대한 공부
4. 각 모델에 대한 내용을 공부
