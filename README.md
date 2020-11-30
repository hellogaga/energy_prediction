# energy_prediction
Based on the input data, a prediction model is trained to predict heat energy demands for the given case. 

## Table of contents
- [energy_prediction](#energy_prediction)
  - [Table of contents](#table-of-contents)
  - [Project Definition, Analysis, and Conclusion](#project-definition-analysis-and-conclusion)
  - [Installation](#installation)
  - [File Structure](#file-structure)
  - [How to Use the Package.](#how-to-use-the-package)
  - [Project Motivation](#project-motivation)
  - [Results](#results)
  - [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)
  
## Project Definition, Analysis, and Conclusion
The detailed definition, analysis and conclusion is given in the [jupyter notebook](Energy_prediction_training.ipynb).

## Installation
The package is based on the following package. The project can be compatible with lower or higher versions of the above packages. However, a detailed test is not carried out. Users might find problems of incompatibilities when executing the code. Please raise a issue when having a problem. <br /> 
- Python: 3.7.9
- pandas: 1.1.3
- numpy:  1.19.2
- matplotlib: 3.3.2
- scipy: 1.5.2
- scikit-learn: 0.23.2
- tensorflow: 2.1.0
- xgboost: 1.2.1
- plotly: 4.12.0
- flask: 1.1.2
<br /> 

## File Structure
- app
  * template
    * master.html (***main page of web app***)
    * go.html  (***classification result page of web app***)
  * run.py  (***Flask file that runs app***)
- data
  * data_impute.csv (***dataset with imputed value***) 
  * Meter_Data.xlsx  (***dataset holds the energy demands***)
  * Public holidays.xlsx (***dataset holds the public holidays***)
  * SMHI.xlsx (***database holds the weather data***)
  * test_data.csv (***merged dataset of weather, public holiday and energy demands***)
- models
  * Final_NN (***Folder that holds the final trained neural network model***)
  * ES_P.csv (***saved parameter for the energy signature model***) 
  * scaler.pkl(***saved standard scaler***)
  * X_col_indice.pkl(***a dictionary that holds the corresponding index and column names***)
  * X_test.csv(***test data set***)
  * y_test.csv(***test data set lables***)
- pics
  * timeasdummy.png(***saved pics***)
  * timeassincos.png(***saved pics***)
- Energy_prediction_training.ipynb(***jupyter notebook used to train the model***)
- Energy_prediction_training.html(***A html copy of the jupyter notebook***)
- README.md

## How to Use the Package. 
- Step 1: Use ```git clone``` or web download to download all the files into a local place in your machine. 
- Step 2: Navigate to downloaded folder. 
- Step 3: Run the jupyter notebook 
```python
 jupyter notebook
```
- Step 4: Navigate to the folder of ***run.py***
- Step 5: Run the web application. The app will be rendered in **a local host**. Can be executed through:
```python
 python run.py 
```
**NOTE1:** The package is tested on a local machine with **Windows 10** as the operation system. Users with Linux or Mac system might have problems in the file locations. Please go to the [jupyter notebook](Energy_prediction_training.ipynb) and revise the file locations accordingly. <br /> 
**NOTE2:** By default, the app will be rendered in a local host. In cloud-based system, such as Udacity's online workspace, revise the line as ```app.run(host='0.0.0.0', port=3001, debug=True)``` in ***run.py***. 

## Project Motivation
This project is the Udacity Data Science Nanodegree Capstone project. The problem originates from industry applications where a simple energy prediction is commonly used. There is great need to develop a new prediction model with better prediction accuracy. 

## Results
The main results of the project is the trained model in the 'models' folder. The main outcome of the project is a web application which can be used to predict energy demands.

## Licensing, Authors, Acknowledgements
The code released subjects to the MIT license. The author appreciates the code structure from [**Udacity**](https://www.udacity.com/).