import json
import plotly
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter, Line, Bar

# define the application
app = Flask(__name__)

# load the scaler/NN model/ES model
recons_scalser = pickle.load(open('../models/scaler.pkl','rb'))
recons_NN = load_model("../models/Final_NN")
ESmodel_p = np.loadtxt('../models//ES_P.csv', delimiter=',')
def ES_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

# reload X_test, 
X_reload_p1 = np.loadtxt('../models/X_test_p1.csv', delimiter=',')
X_reload_p2 = np.loadtxt('../models/X_test_p2.csv', delimiter=',')
X_reload_p3 = np.loadtxt('../models/X_test_p3.csv', delimiter=',')
X_reload = np.append(X_reload_p1, X_reload_p2, axis =0)
X_reload = np.append(X_reload, X_reload_p3, axis =0)
# reload Y_test, X_col_indice
y_reload = np.loadtxt('../models/y_test.csv', delimiter=',')
X_col_reload = pickle.load(open('../models/X_col_indice.pkl','rb'))

# Make predictions on the test dataset
y_pred_reconsES = ES_linear(X_reload[:,X_col_reload['Temperature']], *ESmodel_p)
y_pred_reconsNN = recons_NN.predict(X_reload)

# Define a function that process the input data from webpage forms
def make_input(date, time, Temperature, if_holiday):
    '''
    Read the input from the web form and convert to 
    input that can be understood by the trained model
    
    Parameters
    ----------
    date : string
        date, eg 2015-10-19 
    time : string
        time, eg 14:00 
    Temperature : float 
        Temperature. eg. -5.2
    
    Returns
    -------
    arr_2d : numpy.array
        A 2d array as the input to the prediction model.

    '''
    # Process the input
    input_time = pd.Timestamp(date+' '+time)
    Weekday = input_time.isoweekday()
    Dayofyear = input_time.dayofyear
    Year = input_time.year
    Month = input_time.month
    Day = input_time.day
    Hour = input_time.hour

    if if_holiday =='on' or Weekday in [6,7]:
        if_h = 1
    else:
        if_h = 0

    # make a all zero array
    arr = np.zeros(len(X_col_reload))
    
    # replace values in the matrix according to the location
    arr[X_col_reload['Temperature']] = recons_scalser.transform([[Temperature]])
    arr[X_col_reload['if_Holiday']] = if_h
    arr[X_col_reload['Day of year_' + str(Dayofyear)]]=1
    arr[X_col_reload['Weekday_' + str(Weekday)]]=1
    arr[X_col_reload['Month_' + str(Month)]]=1
    arr[X_col_reload['Day_' + str(Day)]]=1
    arr[X_col_reload['Hour of day_' + str(Hour)]]=1

    arr_2d = np.array([arr])
    return arr_2d

# Render figures at the index page
@app.route('/')
@app.route('/index')
def index(): 
    # create visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=y_reload,
                    y=y_pred_reconsES,
                    mode='markers',
                    name = 'prediction'
                    ),
                Line(
                    x=[0,30000],
                    y=[0,30000],
                    name = 'y=x'
                )
            ],

            'layout': {
                'title': 'Energy Signature Model',
                'yaxis': {
                    'title': "Prediction [Energy]"
                },
                'xaxis': {
                    'title': "True Value [Energy]"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=y_reload,
                    y=y_pred_reconsNN.flatten(),
                    mode='markers',
                    name = 'prediction'
                ),
                Line(
                    x = [0,30000],
                    y = [0,30000],
                    name = 'y=x'
                )     
            ],

            'layout': {
                'title': 'Neural Network Model',
                'yaxis': {
                    'title': "True Value [Energy]"
                },
                'xaxis': {
                    'title': "Prediction [Energy]"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go',methods=["GET", "POST"])
def go():
    # save user input in query
    date = str(request.args.get('date'))#2020-10-11
    time = str(request.args.get('time')) #2020-10-11
    Temperature = float(request.args.get('tem')) # -5.23
    if_holiday = str(request.args.get('holiday')) # 'on' or nothing

    # Make a input array based on the info from webpage
    input_array = make_input(date, time, Temperature, if_holiday)

    # make predictions
    y_pred_reconsES_single = ES_linear(input_array[:,X_col_reload['Temperature']], *ESmodel_p)
    y_pred_reconsNN_single = recons_NN.predict(input_array)

    # Make visuals
    graphs_go = [{              
                'data': [
                    Bar(
                        x=['Energy Signature Model', 'Neural Network Model'],
                        y=[y_pred_reconsES_single[0], y_pred_reconsNN_single.flatten()[0]],
                        name = 'prediction'
                        )
                ],

                'layout': {
                    'title': 'Energy Predictions',
                    'yaxis': {
                        'title': "Prediction [Energy]"
                    },
                    'xaxis': {
                        'title': "Models"
                    }
                }
                }]
    
    # encode plotly graphs in JSON
    ids_go = ["graph-{}".format(i) for i, _ in enumerate(graphs_go)]
    graphJSON_go = json.dumps(graphs_go, cls=plotly.utils.PlotlyJSONEncoder)

    #classification_labels = model.predict([query])[0]
    classification_results = {'date':1,
                              'time':1}

    # This will render the go.html Please see that file. 
    return render_template(
                           'go.html',
                            query = date + ' ' + time + ' Temp: ' + str(Temperature) + '℃',
                            ids=ids_go,
                            graphJSON=graphJSON_go
    )
    #return render_template('master.html', ids=ids, graphJSON=graphJSON)

def main():
    app.run(port=3001, debug=True)


if __name__ == '__main__':
    main()