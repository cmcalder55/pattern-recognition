#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Created On:    2021/10/08
Last Revision: 0000/00/00

<DESCRIPTION>
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
# import yfinance as yf

# metadata
__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2024-Present, Cameron Calder"
__license__=""
__version__= "0.0.0"


## v0 ##

def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        t=.4
        preds[preds > t] = 1
        preds[preds<= t] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        predictions.append(combined)

    return pd.concat(predictions)

## v1 ##

def plotPrice(df,n):
    
    plt.figure()
    
    title = 'Tesla Stock Price for the Past '+str(n)+' Days'
    df.plot(x='Date', y = 'Stock Price', title=title, legend=False)
    
    plt.xticks(rotation=70)
    plt.grid(True)
    
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.show()

 
def formatData(filepath,n):
    df = pd.read_csv(filepath, header=0)
    
    data = {'Date':df['Date'], 'Stock Price':df['Close']}
    df = pd.DataFrame(data)
    
    plotPrice(df,n)
    dates = pd.to_datetime(df['Date'])

    df['Date'] = (dates-dates.min())/np.timedelta64(1,'D')
    
    return  df, dates


def SGDregression(df):
    
    x = df['Date']
    y = df['Stock Price']
    
    X = np.array(x).reshape(-1,1)
    
    xtrain, xtest, ytrain, ytest=train_test_split(X, y, random_state=0, train_size = .75)
    # print(xtest)
    ind = np.concatenate((xtest,xtrain),axis=0)

    # X = np.array(range(30,91)).reshape(-1,1)
    X = scale(X)
    y = scale(y)
    
    xtrain, xtest, ytrain, ytest=train_test_split(X, y, random_state=0, train_size = .75)
    
    sgdr = SGDRegressor()
    sgdr.fit(xtrain, ytrain)
    
    ypred = sgdr.predict(xtest)
    # print(xtest)
    # ypred = sgdr.predict(np.array(range(30,91)).reshape(-1,1))
    
    v = np.concatenate((ypred,ytrain),axis = 0)
    
    dd = pd.DataFrame({'Day':pd.Series(ind.flatten()),
                        'Predicted Price':pd.Series(v.flatten()),
                        'Actual Price': y})
    
    dd = dd.set_index('Day').sort_index().reset_index()
    
    X_new = np.array(range(30,91)).reshape(-1,1)
    y_new = sgdr.predict(scale(X_new))
    
    
    return y, dd,y_new

  
def plotPredictions(df,dates):
    
    actual,dd,y_new = SGDregression(df)
    
    plt.plot(dates,actual,label="Actual")
    plt.plot(dates,dd['Predicted Price'],label="Predicted")
    
    plt.title("Predicted and Actual Stock Price, 30 Days")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    
    plt.xticks(rotation=70)
    
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()
    return dd
    
if __name__ == '__main__':
    
    filepath = "data\\TSLA.csv"   
    
    ## v0 ##
    
    # df = pd.read_csv(filepath,parse_dates=['Date']).set_index('Date')
    
    # # df.plot(y='Close',use_index=True)
    # # record actual stock prices
    # data = df[['Close']]
    # data = data.rename(columns = {'Close':'Actual_Close'})
    # # use rolling method to compare each row to the previous row and assign 1 if greater or 0
    # data["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    # # shift forward one to predict tomorows prices using todays values
    # tsla_prev = df.copy().shift(1)
    
    # predictors = ["Close", "Volume", "Open", "High", "Low"]
    # data = data.join(tsla_prev[predictors]).iloc[1:]
    
    # # Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
    # # model = SGDRegressor()
    # model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
    
    # train = data.iloc[:-100]
    # test = data.iloc[-100:]
    
    # model.fit(train[predictors], train["Target"])
    # predictions = backtest(data, model, predictors)
    
    
    ## v1 ##
    # Predictions for 1 month
    data, dates= formatData(filepath,30)
    dd=plotPredictions(data,dates)
    
    data['Date'] = dates
    df = data.set_index('Date')
    
    forecast = 90
    
    x=dd['Day']
    y=dd['Predicted Price']
    
    a,b = np.polyfit(x,y,1)
    
    plt.scatter(x,y)
    plt.plot(x,a*x+b)
    plt.title("Predicted Stock Price Line of Best Fit")
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    