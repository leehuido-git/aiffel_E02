## 1. 필요한 모듈 import하기
#
import random
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


## 2. 데이터 준비
#
wines = load_wine()

## 3. 데이터 이해하기
#
feature_data = wines.data  #Feature Data 지정하기
label_data = wines.target  #Label Data 지정하기
print(wines.target_names)  #Target Names 출력해 보기
print(wines.DESCR) #데이터 Describe 해 보기


## 4. train, test 데이터 분리
#
x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2, random_state=random.randrange(42))
#random_state는 int 0 and 42을 받는다.
print("data len: {}\ntrain len: {}, test len: {}".format(len(feature_data), len(x_train), len(x_test)))  #데이터 길이 확인


## 5. 다양한 모델로 학습시켜보기
#
models = []
#Decision Tree
decision_tree = DecisionTreeClassifier(random_state=32)
models.append(decision_tree)
decision_tree.fit(x_train, y_train)

#Random Forest
random_forest = RandomForestClassifier(random_state=32)
models.append(random_forest)
random_forest.fit(x_train, y_train)

#Support Vector Machine (SVM)
svm_model = svm.SVC()
models.append(svm_model)
svm_model.fit(x_train, y_train)

#Stochastic Gradient Descent Classifier (SGDClassifier)
sgd_model = SGDClassifier()
models.append(sgd_model)
sgd_model.fit(x_train, y_train)

#Logistic Regression
logistic_model = LogisticRegression()
models.append(logistic_model)
logistic_model.fit(x_train, y_train)


## 6. 모델을 평가해보기
#
y_pred = []
for i in models:
    y_pred.append(i.predict(x_test))

accuracy = [['Decision Tree'], ['Random Forest'], ['SVM'], ['SGDClassifier'], ['Logistic Regression']]
weighted_avg = [['Decision Tree'], ['Random Forest'], ['SVM'], ['SGDClassifier'], ['Logistic Regression']]
for i, y_pred in enumerate(y_pred):
    temp_class_report = classification_report(y_test, y_pred, output_dict=True)
    accuracy[i].append(temp_class_report['accuracy'])
    weighted_avg[i].append(temp_class_report['weighted avg']['f1-score'])

print("weighted avg f1 socre \n:", weighted_avg)
model_name, acc = max(weighted_avg, key=lambda k : k[1])
print("model = {}, weigted_avg = {:.3f}인 사용하면 더 좋을 것 같다.".format(model_name, acc))
#Micro Average(average)는 각각의 TP, FN, FP, TN값들을 모두 합친 total TP, total FN, total FP, total TN값들을 이용해 계산
#Macro Average는 각각의 class에 따라 TP, FN, FP, TN값들을 이용해서 평가 지표를 계산한 후 그 값들의 평균을 사용
#Weighted Average는 각 class에 해당하는 data의 개수에 가중치를 주어 평균
#F1 score는 Recall과 Precision의 조화평균이므로 F1 score에 대해 비교해 보겠습니다.