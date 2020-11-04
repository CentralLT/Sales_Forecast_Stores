# Sales_Forecast_Stores
Predictive Analytics of Retail Sales

# Objective

The goal of this project is to forecast sales for multiple retail branches on a product level for November 2015. We are provided with daily sales data for each branch 
from 2012 to Oktover 2015. The project is hosted by Kaggle. I use this project to compare multiple forecasting models w.r.t. their forecasting performance.
I use: Linear Models, Random Forest, kNN, Ensemble Methods, DNN, CNN, RNN, LSTM models. For more details about the project and results, please have a look at the project 
report file.


# Overview directories and short description

</head>
<body>
        <h1>Directory Tree</h1><p>
        <a></a><br>
        ├── <a>build_features.py</a><br>
        ├── <a>build_model.py</a><br>
        ├── <a>cfg.py</a><br>
        ├── <a</a><br>
        │   ├── <a>output</a><br>
        │   └── <a>raw</a><br>
        ├── <a>downcast.py</a><br>
        ├── <a>main.py</a><br>
        ├── <a>make_dataset.py</a><br>
        ├── <a>project_report</a><br>
        ├── <a>project_report.backup</a><br>
        ├── <a>run.py</a><br>
        └── <a>word_process.py</a><br>
        <br><br>
        </p>
</body>


- Data/output                 <--- Trained and serialized models, model predictions, and output files from feature and model building process
- Data/raw                    <--- Raw Data
- build_features.py          <--- Scripts to turn raw data into features for modeling
- build_model.py             <--- Scripts to train models and then use trained models to make predictions
- cfg.py                     <--- Config file to change model configurations
- downcast.py                <--- Helper function to downcast variables from 64 to 32 bit
- make_dataset.py            <--- Scripts to generate data
- project_report             <--- Generated analysis MarkDown
- requirements.txt           <--- The requirements file stating direct dependencies if a library is developed.
- run.py                     <--- Script to controll workflow (model building, feature engineering, model training & prediction). **Here you start all other scripts!**
- word_process.py            <--- Helper function to produce standardize BOW data



# Data Availability

The data used to support the findings of this project have been deposited in the /Data/raw repository. The data is also available here: 
https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview


# Software Requirements

- Python 3.8
- The file “requirements.txt” lists these dependencies, please run “pip install -r requirements.txt” as the first step. See https://pip.readthedocs.io/en/1.1/requirements.html for further instructions on using the “requirements.txt” file.

# Build with

- d6tflow --- Workflow management tool (comparable to luigi, Airflow)

# Authors

- Robert Schelenz

# License 

This project is licensed under the MIT License - see the LICENSE.md file for details.


