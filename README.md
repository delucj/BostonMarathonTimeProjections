# BostonMarathonTimeProjections
# James DeLuca

The purpose of this project is to develop an improved model for predicting finish time and risk of "hitting the wall" at each of the timing mat splits through the Boston Marathon. This model will be built, tuned and tested on 2015 and 2016 data and validated on 2017 data. The goal will be to develop a model which can produce lower RMSE at every split along the course than the baseline assumption of simply projecting the finish time from the average pace up to the timing mat.

For the runner who may have access to this data and model via smart device, this model may help with race strategy (trying to keep "hitting the wall" risk low while keeping a target time within the confidence interval of the projected time).

For the spectator, this model is intended to help manage social distancing at key observation points along the course. By improving the predictions of when a runner will come past any given point along the course the density of spectators can be more effectively managed to reduce public health risks associated with major marathons such as Boston.

The data used in this project has been downloaded from Kagel. Many thanks to rojour for doing the web-scraping and cleaning to prepare the data sets. With the Active.com decision to abandon race result archives (leaving it up to individual races and race directors to archive their own results) collecting older race result data has become harder of late so I greatly appreciate the effort that prepared the data used in this project. This data is included in this project folder and can also be found here: https://www.kaggle.com/rojour/boston-results

This project is being executed as the final CYO project of the HarvardX PH125.9x Capstone Course on EDX.
