```
Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Final Project: Sparkify
```

#### Customer Churn prediction
The goal of this project is to build a customer churn prediction model using user interaction logs 
with an imaginary music streaming service called Sparkify.

#### Data:
Sparkify is imaginary digital music service similar to Spotify. 
The dataset contains 12GB of user interactions with this fictitious music streaming service.

Data for this project is acquired from Udacity Amazon s3 bucket:
  * https://udacity-dsnd.s3.amazonaws.com/sparkify/sparkify_event_data.json
  * https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json
  
The mini dataset `mini_sparkify_event_data.json` is distributed with this project in the `data` directory.

Run the following wget command to obtain the full dataset: 
  * `wget https://udacity-dsnd.s3.amazonaws.com/sparkify/sparkify_event_data.json`
  
#### Project organization:
  * `src` directory
    * `sparkify.py` -> source code for this project
    * `sparkify.ipynb` -> Jupyter notebook with step by step explanation of our work
  * `report` directory
    * `report.pdf`
  * `output` directory
    * Few output files produced by this program
  * `data`
    * Data directory containing the data obtained to work with this assignment
  * `readme.md`

#### Dependency:
This python program depends on the following modules:
  * time
  * pyspark

#### How to run this program:
  * Navigate to the `src` directory and run one of the files using the command below:
  * `spark-submit Sparkify.py` OR `python Sparkify.py`
