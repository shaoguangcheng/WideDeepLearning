This project is an general implementation of Linear/DNN/Wide&Deep Learning model.

We provide an easy way to build your own machine learning model. What you only need to do is modifying the data_conf.yaml and model_conf.yaml file in conf/ to adapt your problems. 

We also provide several data normalization methods and feature engineering methods. 

Currently, data normalization includes: (1) Z-score normalizer (2) min-max normalizer. For data normalizing, we support processing your data grouped by a given key.

Feature engineering includes: (1) one-hot encoding (2) multi-category feature hashing encoding (3) feature crossing (4) feature bucketizing

Also, several metric methods such as mae、mape、mse、auc、confusion matrix and so on are supported for monitoring your model performance.

We can process data that size exceeds your computer memory. This feature can be turned on if you set option Big_data_mode True in conf/model_conf.yaml. 

