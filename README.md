# market-teller

This project is an attempt to create AI capable of predicting stock prices with recurrent neural networks.

## Overview of the model

We want to compare 2 different approaches to the task of predicting the market situation in this project. Both of the approaches rely only on the past stock data (like technical analysis) and differ mainly in the network architecture and data model. The networks used are:
- Simple, few layers deep, basic neural network; 
- LSTM RNN.

The basic NN is pretty much ready. Input vector for this NN contains data about a set of consecutive days. Each of the days is represented by 5 different values: daily open, high, low and close prices, and volume of stock traded in that day. We have messed around with the input data a little bit and found out that our network performs the best (or rather performs beyond random quess at all), when the data values are smoothed across the time dimension. We've only used so far a 5 days smoothing window which is moved with a single day stride, so there is still some experimenting left to do. 

The network has achieved 68% accuracy in binary classification task of telling whether the smoothed close price of the last day contained in the input vector is greater than the fifth following day smoothed close price or not. The fifth day is used in the comparison, becuase we didn't want the smoothed data to overlap.

The RNN is still work in progress and we don't have any results yet.

## Data

Market-teller uses so far only one, quite big [data set](https://www.quandl.com/product/WIKIP/WIKI/PRICES-Quandl-End-Of-Day-Stocks-Info) acquired from Quandl website.

## What does it need to run?

This project is written purely in python with usage of numpy and tensorflow (ver. 1.0.1) packages.
