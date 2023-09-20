---
layout: default
title: 06. Decision Tree
subtitle: 지능자동화실제 과목
---
-----

[PINBlog Gitea Repository](https://gitea.pinblog.codes/CBNU/06_Decision_Tree)

-----

# 06. 결정 트리(Decision Tree)
- 산업인공지능학과 대학원
    2022254026
        김홍열

# Decision Tree(결정 트리)이란?
분류 및 회귀 문제에 사용되는 지도 학습 알고리즘 중 하나이다. 
트리 구조를 사용하여 결정을 내리거나 값을 예측한다.


# 결정 트리의 주요 개념

### 노드(Node):

* Root Node
    - 트리의 시작점. 전체 데이터 셋을 포함한다.
* Decision Node (Internal Node)
    - 특정 속성에 대한 결정을 내리는 노드.
* Leaf Node (Terminal Node)
    - 최종 결정 또는 예측 값을 포함하는 노드.


### Branch (Edge): 

* 노드 간의 연결, 결정의 결과를 나타낸다.



# 작동 원리

* 분할 기준 선택
    - 데이터를 분할하기 위한 최적의 속성과 그 기준값을 선택한다. 
    - 이를 위해 엔트로피, 지니 불순도, 정보 이득 등의 메트릭을 사용할 수 있다.

* 분할
    - 선택된 속성과 기준값을 기반으로 데이터를 두 개 이상의 하위 집합으로 분할한다.

* 재귀적 분할
    - 각 하위 집합에 대해 1과 2의 과정을 반복한다. 
    - 이는 더 이상 분할이 가능하지 않거나 (모든 항목이 동일한 클래스에 속하거나) 미리 정의된 트리의 깊이나 노드의 최소 크기에 도달할 때까지 계속된다.


# 장점
* 이해하기 쉽다: 트리 구조는 시각적으로 표현하기 쉽고, 비전문가도 쉽게 이해할 수 있다.
* 데이터 전처리가 적다: 결정 트리는 특성의 스케일링이나 정규화가 필요하지 않다.
* 비선형 관계 처리: 결정 트리는 데이터의 비선형 관계를 잘 처리할 수 있다.


# 단점
* 과적합: 결정 트리는 쉽게 과적합될 수 있다. 이는 트리의 깊이를 제한하거나 가지치기(pruning)를 통해 완화될 수 있다.
* 불안정성: 데이터의 작은 변화에도 트리의 구조가 크게 바뀔 수 있다. 이 문제는 랜덤 포레스트와 같은 앙상블 기법을 사용하여 완화될 수 있다.
* 최적화 문제: 최적의 결정 트리를 찾는 것은 NP-완전 문제로 알려져 있다. 따라서 실제로는 근사적인 알고리즘을 사용하여 트리를 구성한다.


---

### 예제 코드 (Random Forest)[¶]()

<details>
<summary>Code View</summary>

<div markdown="1">
  
```c++

//Example 21-1. Creating and training a decision tree

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

void help(char **argv) {
	cout << "\n\n"
		<< "Using binary decision trees to learn to recognize poisonous\n"
		<< "    from edible mushrooms based on visible attributes.\n"
		<< "    This program demonstrates how to create and a train a \n"
		<< "    decision tree using ml library in OpenCV.\n"
		<< "Call:\n" << argv[0] << " <csv-file-path>\n\n"
		<< "\nIf you don't enter a file, it defaults to agaricus-lepiota.data\n"
		<< endl;
}

int main(int argc, char *argv[]) {
	// If the caller gave a filename, great. Otherwise, use a default.
	//
	//const char *csv_file_name = argc >= 2 ? argv[1] : "../mushroom/agaricus-lepiota.data";
	const char *csv_file_name = "agaricus-lepiota.data";
	cout << "OpenCV Version: " << CV_VERSION << endl;
	help(argv);

	// Read in the CSV file that we were given.
	//
	cv::Ptr<cv::ml::TrainData> data_set =
		cv::ml::TrainData::loadFromCSV(csv_file_name, // Input file name
			0, // Header lines (ignore this many)
			0, // Responses are (start) at thie column
			1, // Inputs start at this column
			"cat[0-22]" // All 23 columns are categorical
		);
	// Use defaults for delimeter (',') and missch ('?')
	// Verify that we read in what we think.
	//
	int n_samples = data_set->getNSamples();
	if (n_samples == 0) {
		cerr << "Could not read file: " << csv_file_name << endl;
		exit(-1);
	}
	else {
		cout << "Read " << n_samples << " samples from " << csv_file_name << endl;
	}

	// Split the data, so that 90% is train data
	//
	data_set->setTrainTestSplitRatio(0.90, false);
	int n_train_samples = data_set->getNTrainSamples();
	int n_test_samples = data_set->getNTestSamples();
	cout << "Found " << n_train_samples << " Train Samples, and "
		<< n_test_samples << " Test Samples" << endl;

	// Create a DTrees classifier.
	//
	cv::Ptr<cv::ml::RTrees> dtree = cv::ml::RTrees::create();
	// set parameters
	//
	// These are the parameters from the old mushrooms.cpp code
	// Set up priors to penalize "poisonous" 10x as much as "edible"
	//
	float _priors[] = { 1.0, 10.0 };
	cv::Mat priors(1, 2, CV_32F, _priors);
	dtree->setMaxDepth(8);
	dtree->setMinSampleCount(10);
	dtree->setRegressionAccuracy(0.01f);
	dtree->setUseSurrogates(false /* true */);
	dtree->setMaxCategories(15);
	dtree->setCVFolds(0 /*10*/); // nonzero causes core dump
	dtree->setUse1SERule(true);
	dtree->setTruncatePrunedTree(true);
	dtree->setPriors( priors );
	//dtree->setPriors(cv::Mat()); // ignore priors for now...
	// Now train the model
	// NB: we are only using the "train" part of the data set
	//
	dtree->train(data_set);

	// Having successfully trained the data, we should be able
	// to calculate the error on both the training data, as well
	// as the test data that we held out.
	//
	cv::Mat results;
	float train_performance = dtree->calcError(data_set,
		false, // use train data
		results // cv::noArray()
	);
	std::vector<cv::String> names;
	data_set->getNames(names);
	Mat flags = data_set->getVarSymbolFlags();

	// Compute some statistics on our own:
	//
	{
		cv::Mat expected_responses = data_set->getResponses();
		int good = 0, bad = 0, total = 0;
		for (int i = 0; i < data_set->getNTrainSamples(); ++i) {
			float received = results.at<float>(i, 0);
			float expected = expected_responses.at<float>(i, 0);
			cv::String r_str = names[(int)received];
			cv::String e_str = names[(int)expected];
			cout << "Expected: " << e_str << ", got: " << r_str << endl;
			if (received == expected)
				good++;
			else
				bad++;
			total++;
		}
		cout << "Correct answers: " << (float(good) / total) << " % " << endl;
		cout << "Incorrect answers: " << (float(bad) / total) << "%"
			<< endl;
	}
	float test_performance = dtree->calcError(data_set,
		true, // use test data
		results // cv::noArray()
	);
	cout << "Performance on training data: " << train_performance << "%" << endl;
	cout << "Performance on test data: " << test_performance << " % " << endl;
	return 0;
}

```

</div>

</details>



### 참고[¶]()

- 지능자동화실제 과목, 박태형 교수
- ChatGPT
