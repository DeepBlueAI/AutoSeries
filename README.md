[![Alt text](https://camo.githubusercontent.com/ac37a838f38b505f282164f6d22ff998bbc3a398/68747470733a2f2f7777772e64656570626c756561692e636f6d2f7573722f64656570626c75652f76332f696d616765732f6c6f676f2e706e67)](https://camo.githubusercontent.com/ac37a838f38b505f282164f6d22ff998bbc3a398/68747470733a2f2f7777772e64656570626c756561692e636f6d2f7573722f64656570626c75652f76332f696d616765732f6c6f676f2e706e67)
[![license](https://camo.githubusercontent.com/be29e905b00dad86a9b3e8c1974f3eded6b5ff93/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d47504c253230332e302d677265656e2e737667)](https://github.com/DeepBlueAI/AutoSmart/blob/master/LICENSE)

## Introduction

The 2st place solution for [AutoSeries](https://www.4paradigm.com/competition/autoseries2020).

## Usage

Download the competition's [starting kit](https://autodl.lri.fr/competitions/149#learn_the_details) and run

```
python run_local_test.py --dataset_dir=./data/demo --code_dir=./code_submission
```

You can change the argument `dataset_dir` to other datasets, and change the argument `dataset_dir` to the directory (`code_submission`).

### Dataset

Each dataset containes 5 files: train.data, test.data, test.solution, test_time.data, info.yaml

#### train.data

This is the training data including target variable (regression target). Its column types could be read from info.yaml.
There are 3 data types of features, indicated by "num", "str", and "timestamp", respectively:
• num: numerical feature, a real value
• str: string or categorical features
• timestamp: time feature, an integer that indicates the UNIX timestamp

#### test.data

This is the test data including target variable (regression target). Its column types could be read from info.yaml.

#### test.solution

This is the test solution (extracted from test.data).

#### test_time.data

This is the UNIQUE test timestamp (extracted from test.data).

#### info.yaml

For every dataset, we provide an info.yaml file that contains the important information (meta data).

Here we give details about info.yaml
• time_budget : the time budgets for different methods in user models
• schema : stores data type information of each column
• is_multivariate: whether there are multiple time series.
• is_relative_time: DEPRECATED, not used in this challenge.
• primary_timestamp: UNIX timestamp
• primary_id: a list of column names, identifying uniquely the time series. Note that if is_multivatriate is False, this will be an empty list.
• label: regression target

 

Example:

[![Screen-Shot-2019-11-21-at-21-10-18](https://i.ibb.co/mTmZkYw/Screen-Shot-2019-11-21-at-21-10-18.png)](https://ibb.co/Sy2Yb18)https://camo.githubusercontent.com/8f6aafeb1efa412144302cfad2ae0072e1adc17a/68747470733a2f2f7777772e34706172616469676d2e636f6d2f696d616765732f70632f6175746f6e6c702f64617461312e706e67)

# Contact Us

DeepBlueAI: [1229991666@qq.com](mailto:1229991666@qq.com)