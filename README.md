# A machine learning solution to medical appointment no-show
### Udacity Machine Learning Engineer Nanodegree - Capstone Project

## Getting started

### Installation
This project has been developed on Python 3.7.7.

To install the dependencies, clone the repository, move to the root folder of the repo and run the command:

`pip3 install -r requirements.txt`

#### Troubleshooting and notes
* If running the hyperparameter tuning jobs, the Hyperopt package might need to be installed from source.

* Given the small size, the model has been trained locally on a CPU. GPU support has not been investigated.

### Codebase tour

The project end-to-end can be run by following the two Jupyter notebooks.

The first one, `01. Exploratory Data Analysis` focuses on exploring the dataset and building features to train the machine learning models. This notebook is supported by three modules:
* `utils/plots.py` defines a function used to plot the distribution of categorical and boolean features and their relationship with the target variable. 
* `utils/hypothesis_tests.py` defines a hypothesis test that is run by the plotting function defined in the previous module to understand if any statistically significant difference exists between the values of each distributions and the probability of no-show
* `utils/geocode_neighborhoods.py` uses the Google Maps API to turn the neighborhoods of Vitoria into geographical coordinates and engineer some additional features.
**NOTE**: To run this code is necessary a Google API key enabled for Google Maps. It is not necessary to run the code as the csv file generated by this script is provided in the `/data` folder.

The second notebook - `02. Modeling medical appointments no-show` - focuses on feeding the dataset with the original and engineered features into two predictive models, a Deep and Wide neural network built in Tensorflow (defined in `models/deep_and_wide.py`) and an XGBoost model, evaluating them against the baseline. 

To optimize the results, several iterations of the model have been tried, both in terms of train/validation/test sets and hyperparameters.

The hyperparameter tuning has been performed for both model by using the [Tune package](https://docs.ray.io/en/latest/tune.html) of the [Ray library](https://github.com/ray-project/ray). Both hyperparameters tuning jobs were run with an Asyncronous Successive Halving Scheduler and supported by a Tree-structured Parzen Estimators hyperparameter search algorithm backed by the [Hyperopt package](http://hyperopt.github.io/hyperopt/).

The optimal configuration found for the hyperameters of both models is provided in the `models/dnw_params.json` and `models/xgboost_params.json` files. This allows to skip the tuning process, which can take quite some time, especially for the neural network. The hyperparameters tuning jobs are coded in the `models/nn_hparams_tuning.py` and `models/xgboost_hparams_tuning.py` scripts. Both scripts use `models/load_training_data.py` to load and prepare the data. This last module contains a function that matches 1-to-1 the code at the beginning of the second notebook.

[MLflow](https://mlflow.org/) has been used throughout the project to keep track of the experiments. Also, the [eli5](https://github.com/TeamHG-Memex/eli5/) package has been used to perform a permutation analysis on the models.

## About the project

### Introduction
The recent outbreak of COVID-19 has clearly shown the scarcity of our healthcare resources and has forced us to switch our mindset and actions to be much more conscious of how our own usage of those resources might affect others in need. While the pandemic has amplified the issue, it is not something new to healthcare professionals. 

One of the facets of this issue is patients not showing to medical appointments, therefore not making use of slots that could have been dedicated to others in need. In the US only, medical appointment no shows are believed to have an incidence between 20% and 30% on the total number of appointments, with a study estimating their yearly cost at 150 billion USD for 2006 (Sviokla, John, Bret Schroeder, and Tom Weakland. 2014). 

Scientific studies show that, when applying machine learning and data mining techniques to the studied context, these technologies outperformed the traditional management of the no-shows phenomena (e.g. Srinivas, Sharan, and A. Ravi Ravindran. 2018).

Predicting whether a patient is not going to show up to a medical appointment can be modeled as a binary classification task. Specifically, in this work I will investigate the performances of a deep and wide network in predicting medical appointments no-show. The characteristics of this model will be further discussed in the next chapter. 

### Dataset
Healthcare data is notoriously hard to obtain, as it is shielded by several, really important privacy regulations in place to protect patients. Fortunately, Kaggle offers a great dataset to work with: the [Medical Appointment No Show dataset](https://www.kaggle.com/joniarroba/noshowappointments). 

This dataset is made up of over 110k rows (none of which having missing values) and 14 variables, collecting anonymized data and a binary outcome (show/no-show, our target variable) of medical appointments in public hospitals of Vitoria, Brasil. The data refers to appointments spanning a timeframe of 40 days, between April 29th 2016 and  June 8th 2016, and scheduled over a timeframe of 221 days, from November 10th 2015 to June 8th 2016.

The dataset is explored in the first notebook **01. Exploratory Data Analysis**

### Predicting no-shows
This project focused on investigating the effectiveness of Deep and Wide neural network architectures in predicting medical appointment no-shows. The concept of a deep and wide network was first introduced in the context of recommendation engines by a team of researchers of Google (Cheng, Heng-Tze, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, et al. 2016.). 

The model takes its name from the two main components it can be divided into. 

The first component is a deep neural network, consisting of one or more hidden layers between the input and output layers. This part of the model is in charge of learning abstract, more general patterns (e.g. birds can fly). 

The other component of the model, the wide one, directly connects the input to the output layer, without any intermediate layers. This part of the network is in charge of memorizing certain patterns (e.g. penguins don’t fly). 

Other studies have proven this network architecture to be very effective for different tasks than recommending products, such as electricity theft detection (Zheng, Zibin, Yatao Yang, Xiangdong Niu, Hong-Ning Dai, and Yuren Zhou. 2018).


## Bibliography
* Cheng, Heng-Tze, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, et al. 2016. “Wide &amp; Deep Learning for Recommender Systems.” Proceedings of the 1st Workshop on Deep Learning for Recommender Systems - DLRS 2016. doi:10.1145/2988450.2988454.
* Few S. “Tapping the Power of Visual Perception”, Perceptual Edge. https://www.perceptualedge.com/articles/ie/visual_perception.pdf. 2004
* Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
* Li, Liam, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Moritz Hardt, Benjamin Recht, and Ameet Talwalkar. "Massively parallel hyperparameter tuning." arXiv preprint arXiv:1810.05934 (2018).
* Srinivas, Sharan, and A. Ravi Ravindran. 2018. “Optimizing Outpatient Appointment System Using Machine Learning Algorithms and Scheduling Rules: A Prescriptive Analytics Framework.” Expert Systems with Applications 102: 245–61. doi:10.1016/j.eswa.2018.02.022.
* Sviokla, John, Bret Schroeder, and Tom Weakland. 2014. “How Behavioral Economics Can Help Cure the Health Care Crisis.” Harvard Business Review. August 20. https://hbr.org/2010/03/how-behavioral-economics-can-h.
* Zheng, Zibin, Yatao Yang, Xiangdong Niu, Hong-Ning Dai, and Yuren Zhou. 2018. “Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids.” IEEE Transactions on Industrial Informatics 14 (4): 1606–15. doi:10.1109/tii.2017.2785963.
