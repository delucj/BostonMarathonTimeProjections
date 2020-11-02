##############################################################################################
## "Management of Boston Marathon Finish Line Spectator Density via Machine Learning"
## James DeLuca
## 11/02/2020
## This project is completed as the final CYO machine learning project for HarvardX PH125.9x | Data Science: Capstone
## This script is created to train and test a machine learning algorithm to improve predictions of Boston Marathon finish times
## The models are trained on the results from 2015 and 2016
## The final validation is performed on the 2017 results which are excluded entirely from the training
##
## When you run this script, you will be prompted to select between generating the Random Forest models or downloading previously generated models
## I highly recommned saving time by selecting the download option.
##
## Here are the start and end times when I ran a speed test on the full script and selected to run all calculations
##	> start_time
##	[1] "2020-10-30 20:26:09 EDT"
##	> end_time
##	[1] "2020-11-01 21:54:10 EST"
##
## By comparison, here is the speed test when downloading the models (generated during the full script speed test) from GitHub
##	> start_time
##	[1] "2020-11-02 09:44:06 EST"
##	> end_time
##	[1] "2020-11-02 09:45:44 EST"
## Your processing time may differ depending on internet speed, processing speed and how many of the packages need to be downloaded and installed
##
## By convention, throughout this script I will use single comment symbols "#" to indicated code that has been commented out.
## Where a comment is actually a comment I will preface with double "##" comment symbols
## Where you see "#'" notation at the start of a function, these are the docstring comments which will be referenced by the #?functioncall() method
##
## It may seem odd that distances are in the units of km but paces are in the units of min/mile.
## This mixture of units is not an error; it is a deliberate choice becuase this is the standard method in the United States.
## The most common road race distance is the 5k but most 5km races in the United States do not include km markers on the course, they include mile markers
## At the Boston Marathon, most runners are tracking their paces via GPS watch in minutes per mile but all of the check-points are at metric distances.
## When the Boston Marathon publishes runners paces on their tracking App the pace is in min/mile and the updates are at 5km intervals
##
## Note: this script uses a lot of user defined functions.
## Rather than commenting each time the functions are used the docstring package is used to allow access to the input/output details for most functions
## Use the "?functioncall() method to access the docstrings if desired once the function has been added to your environment
## Additional: functions are not defined at their Point-of-Use. This script has been re-ordered to cluster function definiations at the top of the scrip to clean up the script body
##   If running this script line-by-line please do not skip any of the function definitions as they may be required later in the script
##
##############################################################################################

## Time logging is just for estimating processing speeds
start_time = Sys.time()

#################################
##
## Load Packages
##
#################################

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(plot3D)) install.packages("plot3D", repos = "http://cran.us.r-project.org")
if(!require(docstring)) install.packages("docstring", repos = "http://cran.us.r-project.org")
if(!require(svDialogs)) install.packages("svDialogs", repos = "http://cran.us.r-project.org")
if(!require(ggsci)) install.packages("ggsci", repos = "http://cran.us.r-project.org")

library(dplyr)
library(ggplot2)
library(data.table)
library(tidyverse)
library(caret)
library(ggpubr)
library(stringr)
library(randomForest)
library(plot3D)
library(docstring)
library(svDialogs)
library(ggsci)


#################################
##
## User Prompt
## RF tuning and training can be slow, there are 8 stages along the race
## Random forest models are trained at each stage with node size and tree count optimized
## I highly recommend running this chunk and selecting "Yes" so that you can download the pre-generated models
## Regenerating all models may take tens of hours
##
#################################

download_rfs <- "yes"    # This defaults the download_rfs variable. If you are running this script line by line you can send this line and skip the dialog message below

download_rfs <- dlg_message(title = "Use Pre-Generated RF Objects",
                            message = c('Do you want to skip Random Forest Model tuning and training?','',
                                        'This project contains random forest models at 9 splits along the Boston Marathon course.',
                                        'Tuning and training all of these models will take an estimated 50 hours.','',
                                        'This training has already been completed and trained models can be downloaded from GitHub.','',
                                        'Select "Yes" to skip RF training blocks and download the completed objects.',
                                        'Select "No" to run all calculationd directly.','',
                                        'I highly recommend that you select the pre-generated objects.'),
                            type = "yesno", gui = .GUI)$res


#################################
##
## General Use Functions
## This section contains relatively generalized functions which may be applicable elsewhere
##
#################################

hms_to_min <- function(race_time){
  #' This function takes a timestamp in the hh:mm:ss or h:m:s format and returns a time in minutes

  indicies = str_locate_all(pattern =fixed(':'),race_time)
  hours = as.numeric(substr(race_time,0,indicies[[1]][1]-1))
  minutes = as.numeric(substr(race_time,indicies[[1]][1]+1,indicies[[1]][2]-1))
  seconds = as.numeric(substr(race_time,indicies[[1]][2]+1,indicies[[1]][2]+2))
  
  time = (60 * hours) + minutes + (seconds / 60)
  return(time)
}

convert_min_to_racetime <- function(time){
  #' This function converts a race time from the units of minutes to a text string in the format h:mm:ss

  hours = floor(time / 60)
  minutes = floor((time/60 - floor(time/60)) * 60)
  seconds = round((time - floor(time))*60,0)

  # Pad the minutes and seconds with leading zeros so that we don't get results like 3:4:7
  min_disp = str_sub(paste("00",minutes,sep=""),start = -2)
  sec_disp = str_sub(paste("00",seconds,sep=""),start = -2)

  return(paste(hours,":",min_disp,":",sec_disp,sep=""))
}

read_csv_from_github <- function(githubURL,file_name) {
  #' This function takes a URL and a *.csv file name and downloads then reads the csv into a data frame
  
  download.file(githubURL,file_name,method = 'curl')
  read.csv(file_name)
}

read_rds_from_github <- function(githubURL,file_name){
  #' This function takes a URL and a *.rds file name and downloads and then reads the file into an R object
  
  download.file(githubURL,file_name)
  readRDS(file_name)
}

RMSE <- function(acutal, predicted){
  #' This function takes two equal length vectors and calculates the Root Mean Square Error of one as a predictor of the other
  #' The order of the vectors is not important; the returned RMSE is in the same units as the input vectors
  
  rmse_dat = data.frame(Actual = acutal,Predicted = predicted)
  
  # Filter out any rows where there is missing data
  rmse_dat = rmse_dat %>%
    filter(!is.na(Actual)) %>%
    filter(!is.na(Predicted))
  
  rmse = sqrt(mean((rmse_dat$Actual - rmse_dat$Predicted)^2))
  return(rmse)
}

smart_set_seed <- function(seed_value){
  #' The input is a random seed value
  #' This function will determine what version of R is in use and set the seed with the appropriate method
  
  if (as.numeric(substr(paste(R.Version()$major,".",R.Version()$minor,sep=""),1,3)) > 3.5) {
    set.seed(seed_value, sample.kind="Rounding") } else {
      set.seed(seed_value)}
}

solve_quadratic <- function(y_data,x_data){
  #' This is a closed form solution to solve for the coefficients of a 2nd order polynomial fit
  #' The input is a pair of vectors, the dependent variable vector is supplied as y-data first
  #' The second input is the predictor vector
  #' This function will return the x_data value where the derivative of the polynomial fit is 0

  # In the context of this project I'm using this for finding minimum points by passing the smallest value and its neighbors
  # The same function works for local maxima if the maximum is passed with its neighbors
  
  n = nrow(data.frame(y_data))
  xbar = sum(x_data) / n
  ybar = sum(y_data) / n
  x2bar = sum(x_data^2) / n
  
  Sxx = sum((x_data - xbar)^2)
  Sxy = sum((x_data - xbar) * (y_data - ybar))
  Sxx2 = sum((x_data - xbar) * (x_data^2 - x2bar))
  Sx2x2 = sum((x_data^2 - x2bar)^2)
  Sx2y = sum((x_data^2 - x2bar) * (y_data - ybar))
  
  # Coefficients B0, B1, B2 (form y = B0 + B1*x + B2*x^2)
  B2 = (Sx2y*Sxx - Sxy*Sxx2) / ((Sxx*Sx2x2) - (Sxx2**2))
  B1 = (Sxy*Sx2x2 - Sx2y*Sxx2) / ((Sxx*Sx2x2) - (Sxx2**2))
  B0 = ybar - B1*xbar - B2*x2bar
  
  # Derivative: dy/dx = 2*B2*x + B1
  # dy/dx = 0 @ the inflection
  # 0 = 2B2*x + B1 so (-B1 / 2*B2) = x
  x_for_inflection = -B1 / (2 * B2)

  return(x_for_inflection)
}

make_train_test_sets <- function(data,test_size,seed_value){
  #' This purpose of this function is to get training and testing sets from a body of data
  #' The input is a dataframe, the fraction of the data to put into the testing set and a random seed value
  #' The function will return a list with the training set at [[1]] and the test set at [[2]]
  
  smart_set_seed(seed_value)
  test_index = createDataPartition(y = data$Final_Time, times = 1, p = test_size, list = FALSE)
  
  train = data[-test_index,]
  test = data[test_index,]
  
  # R does not support tuple unpacking but using a list to return multiple objects seems relatively equivalent
  train_test_sets = list(train,test)
  return(train_test_sets)
}


#########################################################
##
## Boston Marathon Data Cleaning and Processing Functions
## This section contains the pre-processing functions needed to clean and prepare Boston Marathon results data
##
#########################################################

pre_process_data <- function(boston_data,race_year){
  #' This function is a container for all the Boston Marathon data cleaning functions
  #' The input is a raw Boston Marathon results dataframe which will be returned ready for use with the models in this project

  boston_data = clean_results_data(boston_data)
  boston_data = add_runner_wave(boston_data,race_year)
  boston_data = determine_cutoff_pace(boston_data,race_year)
  boston_data = add_expected_bq_times(boston_data)
  boston_data = add_bq_normalized_times(boston_data)
  boston_data = add_naive_finish_times(boston_data)
  boston_data = add_pace_since_last_split(boston_data)
  boston_data = add_normliazed_paces(boston_data)
  boston_data = filter_bad_columns(boston_data)
  
  return(boston_data)
}

clean_results_data <- function(boston_data){
  #' This function converts the Marathon results from h:m:s format to minutes and filters out corrupted data
  #' Elite (professional) results are removed since they are not representative of the general race population
  #' The only input required is a Boston Marathon results dataframe; the mutated dataframe is returned

  boston_data = boston_data %>%
    mutate(KM05_Time = hms_to_min(X5K),
           KM10_Time = hms_to_min(X10K),
           KM15_Time = hms_to_min(X15K),
           KM20_Time = hms_to_min(X20K),
           Half_Time = hms_to_min(Half),
           KM25_Time = hms_to_min(X25K),
           KM30_Time = hms_to_min(X30K),
           KM35_Time = hms_to_min(X35K),
           KM40_Time = hms_to_min(X40K),
           Final_Time = hms_to_min(Official.Time)) %>%
    filter(!is.na(KM05_Time) &
             !is.na(KM10_Time) &
             !is.na(KM15_Time) &
             !is.na(KM20_Time) &
             !is.na(Half_Time) &
             !is.na(KM25_Time) &
             !is.na(KM30_Time) &
             !is.na(KM35_Time) &
             !is.na(KM40_Time) &
             !is.na(Final_Time)) %>%
    mutate(Bib = as.numeric(Bib)) %>%
    filter(!is.na(Bib) & Bib > 100)
  ## The B.A.A. officially starts qualified (non-professional or sponsored) runners at Bib number 101, some elite bibs are non-numeric
  
  return(boston_data)
}

add_runner_wave <- function(boston_data,race_year){
  #' This function will download the wave-information by year and assign the appropriate wave to each runner
  #' The input required is a Boston Marathon results dataframe and the year of the race
  #' The function will return the dataframe with wave specific information for each runner
  #' The wave_info dataframe is assinged with the global variable assignment opperator "<<-" to make it available in other functions without requiring that we return it from this function
  
  ## The wave number and wave-cutoff information is available on the BAA site but I put it into a csv just in case the BAA changes the URL
  wave_info <<- read_csv_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/marathon_wave_information.csv","marathon_wave_information.csv")
  
  ## Since each year the data may be different we'll filter by the race year
  this_year_wave_data = wave_info %>%
    filter(Year == race_year) %>%
    mutate(Wave = as.factor(Wave))
  
  ## From the bib number and the wave data we can assign each runner to one of the 4 waves
  boston_data = boston_data %>%
    filter(Bib > min(this_year_wave_data$Min.Bib.Number)) %>%
    mutate(Wave = as.factor(ifelse(Bib < this_year_wave_data$Min.Bib.Number[2],1,
                                   ifelse(Bib < this_year_wave_data$Min.Bib.Number[3],2,
                                          ifelse(Bib < this_year_wave_data$Min.Bib.Number[4],3,4)))))
  
  ## The wave is an indication of runner qualification time and the fastest and slowest qualification times in each wave.
  this_year_wave_data = this_year_wave_data %>%
    select(Wave,Min.Qual.Time,Max.Qual.Time,Cutoff)
  
  boston_data = left_join(boston_data,this_year_wave_data,by="Wave")
  return(boston_data)
}

determine_cutoff_pace <- function(boston_data,race_year){
  #' This function will download and merge in the qualification standards for the next Boston Marathon
  #' The input is a Boston Marathon results dataframe and the race year
  #' The function will return the dataframe with the times that each runner would need to run to requalify
  #' The qual_info dataframe is assinged with the global variable assignment opperator "<<-" to make it available in other functions without requiring that we return it from this function
  
  ## The qualification information is available on the BAA site but I put it into a csv just in case the BAA changes the URL to this information
  qual_info <<- read_csv_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/marathon_qualification_standards.csv","marathon_qualification_standards.csv")
  
  ## Since each year the data may be different we'll filter by the race year
  this_year_wave_data = wave_info %>%
    filter(Year == race_year)
  cutoff_modifier = mean(this_year_wave_data$Cutoff)
  
  ## Everyone's age is incremented by 1 since the qualification standards are based on what your age will be on the next race year not your age on this race day
  next_year_qual_time = ifelse(boston_data$M.F == "M",
                               ifelse(boston_data$Age + 1 < qual_info$Max.Age[1],qual_info$Men[1],
                                      ifelse(boston_data$Age + 1 < qual_info$Max.Age[2],qual_info$Men[2],
                                             ifelse(boston_data$Age + 1 < qual_info$Max.Age[3],qual_info$Men[3],
                                                    ifelse(boston_data$Age + 1 < qual_info$Max.Age[4],qual_info$Men[4],
                                                           ifelse(boston_data$Age + 1 < qual_info$Max.Age[5],qual_info$Men[5],
                                                                  ifelse(boston_data$Age + 1 < qual_info$Max.Age[6],qual_info$Men[6],
                                                                         ifelse(boston_data$Age + 1 < qual_info$Max.Age[7],qual_info$Men[7],
                                                                                ifelse(boston_data$Age + 1 < qual_info$Max.Age[8],qual_info$Men[8],
                                                                                       ifelse(boston_data$Age + 1 < qual_info$Max.Age[9],qual_info$Men[9],
                                                                                              ifelse(boston_data$Age + 1 < qual_info$Max.Age[10],qual_info$Men[10],
                                                                                                     qual_info$Men[11])))))))))),
                               ifelse(boston_data$boston_data$Age + 1 < qual_info$Max.Age[1],qual_info$Women[1],
                                      ifelse(boston_data$Age + 1 < qual_info$Max.Age[2],qual_info$Women[2],
                                             ifelse(boston_data$Age + 1 < qual_info$Max.Age[3],qual_info$Women[3],
                                                    ifelse(boston_data$Age + 1 < qual_info$Max.Age[4],qual_info$Women[4],
                                                           ifelse(boston_data$Age + 1 < qual_info$Max.Age[5],qual_info$Women[5],
                                                                  ifelse(boston_data$Age + 1 < qual_info$Max.Age[6],qual_info$Women[6],
                                                                         ifelse(boston_data$Age + 1 < qual_info$Max.Age[7],qual_info$Women[7],
                                                                                ifelse(boston_data$Age + 1 < qual_info$Max.Age[8],qual_info$Women[8],
                                                                                       ifelse(boston_data$Age + 1 < qual_info$Max.Age[9],qual_info$Women[9],
                                                                                              ifelse(boston_data$Age + 1 < qual_info$Max.Age[10],qual_info$Women[10],
                                                                                                     qual_info$Women[11])))))))))))
  
  ## Each year too many qualifiers try to sign up for the race so runners know that they need to beat their qualification standard by at least the cutoff that they beat to get accepted
  ## I do not know if proximity to a cutoff time will impact runner performance so I'm going to add in this modified feature
  boston_data = boston_data %>% mutate(BQ_Cutoff = (next_year_qual_time - cutoff_modifier))
  
  return(boston_data)
}

add_expected_bq_times <- function(boston_data){
  #' This function adds a check point time that each runner should be at to qualify for the next Boston with even splits
  #' The input is a Boston marathon results data frame which has already been processed with determine_cutoff_pace()
  #' The function returns the mutated dataframe
  
  boston_data = boston_data %>%
    mutate(KM05_BQ_Time = BQ_Cutoff * 3.10686 / 26.2188,
           KM10_BQ_Time = BQ_Cutoff * 6.21371 / 26.2188,
           KM15_BQ_Time = BQ_Cutoff * 9.32057 / 26.2188,
           KM20_BQ_Time = BQ_Cutoff * 12.4274 / 26.2188,
           Half_BQ_Time = BQ_Cutoff * 13.1094 / 26.2188,
           KM25_BQ_Time = BQ_Cutoff * 15.5343 / 26.2188,
           KM30_BQ_Time = BQ_Cutoff * 18.6411 / 26.2188,
           KM35_BQ_Time = BQ_Cutoff * 21.748 / 26.2188,
           KM40_BQ_Time = BQ_Cutoff * 24.8548 / 26.2188)
  
  return(boston_data)
}

add_bq_normalized_times <- function(boston_data){
  #' This function adds features for what % faster or slower each runner is at each check-point than their qualification pace
  #' The input is a boston marathon results data frame which has already been processed with add_expected_bq_times()
  #' The function returns the mutated dataframe

  boston_data = boston_data %>%
    mutate(KM05_BQ_Nrm = 100*(KM05_Time - KM05_BQ_Time) / KM05_BQ_Time,
           KM10_BQ_Nrm = 100*(KM10_Time - KM10_BQ_Time) / KM10_BQ_Time,
           KM15_BQ_Nrm = 100*(KM15_Time - KM15_BQ_Time) / KM15_BQ_Time,
           KM20_BQ_Nrm = 100*(KM20_Time - KM20_BQ_Time) / KM20_BQ_Time,
           Half_BQ_Nrm = 100*(Half_Time - Half_BQ_Time) / Half_BQ_Time,
           KM25_BQ_Nrm = 100*(KM25_Time - KM25_BQ_Time) / KM25_BQ_Time,
           KM30_BQ_Nrm = 100*(KM30_Time - KM30_BQ_Time) / KM30_BQ_Time,
           KM35_BQ_Nrm = 100*(KM35_Time - KM35_BQ_Time) / KM35_BQ_Time,
           KM40_BQ_Nrm = 100*(KM40_Time - KM40_BQ_Time) / KM40_BQ_Time)
  
  return(boston_data)
}

add_naive_finish_times <- function(boston_data){
  #' The B.A.A. runner tracking app projects the finish time for each runner at each split to be their average pace to that point extended out over the full course
  #' This function adds the "Naive" prediction at each split. This is the baseline model that we're trying to improve
  #' The function returns the mutated dataframe

    boston_data = boston_data %>%
      mutate(naive_KM05_Finish = KM05_Time * 26.2188 / 3.10686,
             naive_KM10_Finish = KM10_Time * 26.2188 / 6.21371,
             naive_KM15_Finish = KM15_Time * 26.2188 / 9.32057,
             naive_KM20_Finish = KM20_Time * 26.2188 / 12.4274,naive_Half_Finish = Half_Time * 26.2188 / 13.1094,
             naive_KM25_Finish = KM25_Time * 26.2188 / 15.5343,
             naive_KM30_Finish = KM30_Time * 26.2188 / 18.6411,
             naive_KM35_Finish = KM35_Time * 26.2188 / 21.748,
             naive_KM40_Finish = KM40_Time * 26.2188 / 24.8548)

    return(boston_data)
}

add_pace_since_last_split <- function(boston_data){
  #' This function adds the pace that each runner ran from the last check-point to the current check-point
  #' These features are to let the model learn what trends in a runner's pace mean towards their finish time
  #' The input is a Boston Marathon results dataframe
  #' The function returns the mutated dataframe
  
  boston_data = boston_data %>%
    mutate(PaceAt5K = KM05_Time / 3.10686,
           PaceAt10K = (KM10_Time - KM05_Time) / (6.21371 - 3.10686),
           PaceAt15K = (KM15_Time - KM10_Time) / (9.32057 - 6.21371),
           PaceAt20K = (KM20_Time - KM15_Time) / (12.4274 - 9.32057),
           PaceAtHalf = (Half_Time - KM20_Time) / (13.1094 - 12.4274),
           PaceAt25K = (KM25_Time - Half_Time) / (15.5343 - 13.1094),
           PaceAt30K = (KM30_Time - KM25_Time) / (18.6411 - 15.5343),
           PaceAt35K = (KM35_Time - KM30_Time) / (21.748 - 18.6411),
           PaceAt40K = (KM40_Time - KM35_Time) / (24.8548 - 21.748))
  
  return(boston_data)
}

add_normliazed_paces <- function(boston_data){
  #' For each split after 10km this function will add the ratio of the runners most recent pace to their average pace over the first 10km
  #' This function is intended to try to simplify the random forest regression trees that look for slowdown effects
  #' The input is a Boston Marathon results dataframe, the return is the mutated dataframe
  #' The function returns the mutated dataframe

    boston_data = boston_data %>%
      mutate(NrmPace_15K = 6.21371 * PaceAt15K / KM10_Time,
             NrmPace_20K = 6.21371 * PaceAt20K / KM10_Time,
             NrmPace_Half = 6.21371 * PaceAtHalf / KM10_Time,
             NrmPace_25K = 6.21371 * PaceAt25K / KM10_Time,
             NrmPace_30K = 6.21371 * PaceAt30K / KM10_Time,
             NrmPace_35K = 6.21371 * PaceAt35K / KM10_Time,
             NrmPace_40K = 6.21371 * PaceAt40K / KM10_Time,)
    
    return(boston_data)
}

add_mid_race_miss <- function(boston_data){
  #' We build a series of linear regression models to predict when each runner will get to the next check-point based on the last check-point
  #' The runner's residual to that prediction is important in predicting the final time
  #' This function takes a dataframe and gets the runner's residuals and ratios at each check-point for use in the random forest models
  #' The function returns the mutated dataframe
  
  ## The additive random forest models look at the residuals (actual - linear prediction) from check-point to check-point
  boston_data$MissAt10k_add = boston_data$KM10_Time - predict(pred10k_lm,newdata = boston_data)
  boston_data$MissAt15k_add = boston_data$KM15_Time - predict(pred15k_lm,newdata = boston_data)
  boston_data$MissAt20k_add = boston_data$KM20_Time - predict(pred20k_lm,newdata = boston_data)
  boston_data$MissAt_Half_add = boston_data$Half_Time - predict(pred_half_lm,newdata = boston_data)
  boston_data$MissAt25k_add = boston_data$KM25_Time - predict(pred25k_lm,newdata = boston_data)
  boston_data$MissAt30k_add = boston_data$KM30_Time - predict(pred30k_lm,newdata = boston_data)
  boston_data$MissAt35k_add = boston_data$KM35_Time - predict(pred35k_lm,newdata = boston_data)
  boston_data$MissAt40k_add = boston_data$KM40_Time - predict(pred40k_lm,newdata = boston_data)
  
  ## The multiplicative random forest models look at the ratios (actual / linear prediction) from check-point to check-point
  boston_data$MissAt10k_mult = boston_data$KM10_Time / predict(pred10k_lm,newdata = boston_data) 
  boston_data$MissAt15k_mult = boston_data$KM15_Time / predict(pred15k_lm,newdata = boston_data)
  boston_data$MissAt20k_mult = boston_data$KM20_Time / predict(pred20k_lm,newdata = boston_data)
  boston_data$MissAt_Half_mult = boston_data$Half_Time / predict(pred_half_lm,newdata = boston_data)
  boston_data$MissAt25k_mult = boston_data$KM25_Time / predict(pred25k_lm,newdata = boston_data)
  boston_data$MissAt30k_mult = boston_data$KM30_Time / predict(pred30k_lm,newdata = boston_data)
  boston_data$MissAt35k_mult = boston_data$KM35_Time / predict(pred35k_lm,newdata = boston_data)
  boston_data$MissAt40k_mult = boston_data$KM40_Time / predict(pred40k_lm,newdata = boston_data)
  
  return(boston_data)
}

filter_bad_columns <- function(boston_data){
  #' The web-scrape sometimes returns some un-identified columns
  #' This function strips them out so that we can do an effective rbind() when creating combined training/testing sets
  #' The function takes in a dataframe of Boston Marathon results and returns the data-frame without the bad columns
  
  ## It only really matters that we do this for the 2015 and 2016 results so that we can bind the dataframes to make composite training and testing sets
  ## "X" is present in both 2015 but only 2016 has "X.1". These features do not appear to hold any useful information but can get in the way of the bind so I'm throwing them away
  ## I didn't see an "X.2" feature but the code is trivial and the time impact is near zero to have the check and removal so I added the check as buffer

  if("X" %in% colnames(boston_data)) {
    boston_data = boston_data %>%
      select(!X)
  }
  if("X.1" %in% colnames(boston_data)) {
    boston_data = boston_data %>%
      select(!X.1)
  }
  if("X.2" %in% colnames(boston_data)) {
    boston_data = boston_data %>%
      select(!X.2)
  }
  return(boston_data)
}

get_naive_performance <- function(boston_data){
  #' This function takes a dataframe of pre-processed Boston Marathon results
  #' This function will return a dataframe with the RMS at each split for the BAA prediction
  #' The predictions all just assume constant pace over the entire race equal to the pace at the check-point
  
  naive_rmse_05km = boston_data %>% filter(!is.na(naive_KM05_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM05_Finish))
  naive_rmse_10km = boston_data %>% filter(!is.na(naive_KM10_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM10_Finish))
  naive_rmse_15km = boston_data %>% filter(!is.na(naive_KM15_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM15_Finish))
  naive_rmse_20km = boston_data %>% filter(!is.na(naive_KM20_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM20_Finish))
  naive_rmse_half = boston_data %>% filter(!is.na(naive_Half_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_Half_Finish))
  naive_rmse_25km = boston_data %>% filter(!is.na(naive_KM25_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM25_Finish))
  naive_rmse_30km = boston_data %>% filter(!is.na(naive_KM30_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM30_Finish))
  naive_rmse_35km = boston_data %>% filter(!is.na(naive_KM35_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM35_Finish))
  naive_rmse_40km = boston_data %>% filter(!is.na(naive_KM40_Finish)) %>%
    summarize(ModRMSE = RMSE(Final_Time,naive_KM40_Finish))
  
  # Bind the results to return as a dataframe
  # The distance vector is the distance in KM of each split
  naive_performance = data.frame(
    cbind(Distance = as.numeric(c(5,10,15,20,21.0975,25,30,35,40)),
          Naive.RMSE = as.numeric(c(naive_rmse_05km,naive_rmse_10km,naive_rmse_15km,
                                    naive_rmse_20km,naive_rmse_half,naive_rmse_25km,
                                    naive_rmse_30km,naive_rmse_35km,naive_rmse_40km))))
  
  return(naive_performance)
}

get_linear_performance <- function(boston_data){
  #' This function takes a dataframe of pre-processed Boston Marathon results
  #' This function will return a dataframe with the RMSE at each split for the linear model predictions
  #' All predictions are multi-variable linear regression model predictions trained on 2015 and 2016 data
  
  lm_rmse_05km = RMSE(boston_data$Final_Time,predict(mod_lm_05,boston_data))
  lm_rmse_10km = RMSE(boston_data$Final_Time,predict(mod_lm_10,boston_data))
  lm_rmse_15km = RMSE(boston_data$Final_Time,predict(mod_lm_15,boston_data))
  lm_rmse_20km = RMSE(boston_data$Final_Time,predict(mod_lm_20,boston_data))
  lm_rmse_half = RMSE(boston_data$Final_Time,predict(mod_lm_Half,boston_data))
  lm_rmse_25km = RMSE(boston_data$Final_Time,predict(mod_lm_25,boston_data))
  lm_rmse_30km = RMSE(boston_data$Final_Time,predict(mod_lm_30,boston_data))
  lm_rmse_35km = RMSE(boston_data$Final_Time,predict(mod_lm_35,boston_data))
  lm_rmse_40km = RMSE(boston_data$Final_Time,predict(mod_lm_40,boston_data))
  
  # Bind the results to return as a dataframe
  # The distance vector is the distance in KM of each split
  linear_performance = data.frame(
    cbind(Distance = as.numeric(c(5,10,15,20,21.0975,25,30,35,40)),
          Linear.RMSE = as.numeric(c(lm_rmse_05km,lm_rmse_10km,lm_rmse_15km,
                                     lm_rmse_20km,lm_rmse_half,lm_rmse_25km,
                                     lm_rmse_30km,lm_rmse_35km,lm_rmse_40km))))
  
  return(linear_performance)
}

get_rf_performance <- function(boston_data){
  #' This function takes a dataframe of pre-processed Boston Marathon results
  #' This function will return a dataframe with the RMSE at each split for the linear regression and regression tree ensemble model predictions
  #' The predict_rf* functions take care of mixing the two Random Forest models with the linear regression model
  
  rf_rmse_05km = RMSE(boston_data$Final_Time,predict_rf_05km(boston_data))
  rf_rmse_10km = RMSE(boston_data$Final_Time,predict_rf_10km(boston_data))
  rf_rmse_15km = RMSE(boston_data$Final_Time,predict_rf_15km(boston_data))
  rf_rmse_20km = RMSE(boston_data$Final_Time,predict_rf_20km(boston_data))
  rf_rmse_half = RMSE(boston_data$Final_Time,predict_rf_half(boston_data))
  rf_rmse_25km = RMSE(boston_data$Final_Time,predict_rf_25km(boston_data))
  rf_rmse_30km = RMSE(boston_data$Final_Time,predict_rf_30km(boston_data))
  rf_rmse_35km = RMSE(boston_data$Final_Time,predict_rf_35km(boston_data))
  rf_rmse_40km = RMSE(boston_data$Final_Time,predict_rf_40km(boston_data))
  
  # Bind the results to return as a dataframe
  # The distance vector is the distance in KM of each split
  rf_performance = data.frame(
    cbind(Distance = as.numeric(c(5,10,15,20,21.0975,25,30,35,40)),
          RF.RMSE = as.numeric(c(rf_rmse_05km,rf_rmse_10km,rf_rmse_15km,
                                 rf_rmse_20km,rf_rmse_half,rf_rmse_25km,
                                 rf_rmse_30km,rf_rmse_35km,rf_rmse_40km))))
  
  return(rf_performance)
}

setup_for_rf_tune_for <- function(){
  #' This function will setup the nt (number of trees) and reset the rmses vector to an empty vector
  #' This function is used to prepare for the nodesize tuning loop for the Random Forest models used in this project
  #' This function edits global variables so no return is necessary
  
  nt <<- 75
  rmses <<- vector(mode="numeric",length=0)
}

setup_for_rf_tune_while <- function(nodes,rmses){
  #' This function sets old_rmse, old_nt and new_rmse values based on the output of the nodesize tuning for loops
  #' This function is used to initialize these values for use in the while loop that grows the random forest to an optimized size
  #' The input is the nodesize vector and the vector of RMSEs that correspond to those node sizes
  #' This function edits global variables so no return is necessary
  
  ## Node size is optimized from the for loop. I'm using the solve_quadratic function to find the best nodesize
  ## If the best position is at a boundary I just use that boundary
  if(nodes[which.min(rmses)] == max(nodes)) {
    ns = max(nodes)
  } else if (nodes[which.min(rmses)] == min(nodes)) {
    ns = min(nodes)
  } else {
    ns <<- round(solve_quadratic(data.frame(c(rmses[which.min(rmses)-1],rmses[which.min(rmses)],rmses[which.min(rmses)+1])),
                                 data.frame(c(nodes[which.min(rmses)-1],nodes[which.min(rmses)],nodes[which.min(rmses)+1]))),0)
  }
  
  old_rmse <<- min(rmses)	# We will start with trying to optimize the model coming out of node-size tuning
  old_nt <<- 75			# We'll always start at 75 trees
  new_rmse <<- 0			# This ensures we will always try at least one iteration of the while loop
}

#########################################################
##
## Training Data Download
## I have placed copies of the data collected from Kaggle in a GitHub repository
## This is to ensure that the data being pulled for this analysis does not change over time
##
#########################################################

bm_2015_results <- read_csv_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/marathon_results_2015.csv","marathon_results_2015.csv")
bm_2016_results <- read_csv_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/marathon_results_2016.csv","marathon_results_2016.csv")

## Set aside an unprocessed head from one of the results files for the report
bm_2015_knit <- head(bm_2015_results)


#########################################################
##
## Pre-Process the Training Data
##
#########################################################

## This just calls the function which will call all the necessary cleaning and processing functions
bm_2015_results <- pre_process_data(bm_2015_results,2015)
bm_2016_results <- pre_process_data(bm_2016_results,2016)

## Here is what the front end of the race looks like:
bm_2015_knit %>% select(Bib,Name,Age,M.F,Country,X5K,Half,X35K,Official.Time,Pace)

## Here is what the race looks like on average, illustrating how different the race winners are from the rest of us
bm_2015_results %>% group_by(Country) %>%
  summarize(Runners = length(Bib),
            Mean.Finish = convert_min_to_racetime(mean(Final_Time)),
            Mean.Age = round(mean(Age),1) )%>%
  arrange(desc((Runners))) %>% filter(Runners > 100)


#########################################################
##
## Data Visualization Functions
## I collected most of the data visualizations into this block
## The purpose of making thes functions is so that I can easily pass in new data without re-coding the formatting
##
#########################################################

plot_finish_gender_distribution <- function(boston_data,race_year){
  #' This function will return a plot object that will visualize the distribution of race finish times by sex
  #' The input is a Boston Marathon data frame, it is not necessary that it be pre-processed
  
  # Calculate the mean and standard deviation for the plot title
  mean_finish <- round(mean(boston_data$Final_Time),2)
  finish_sigma <- round(sd(boston_data$Final_Time),2)
  
  boston_data %>%
    mutate(M.F = ifelse(M.F == "F","Female","Male")) %>%
    ggplot() +
    geom_histogram(aes(x=Final_Time,fill=M.F),color="black",binwidth=5) +
    xlab("Actual Finish Time (minutes)") +
    ggtitle(paste("Distribution of Finish Times by Sex |",race_year,
                  "Mean =",mean_finish,"Sigma =",finish_sigma)) +
    guides(fill=guide_legend(title="Male/Female")) +
    xlim(c(150,400)) + scale_fill_igv()
}
## The general distribution moves to the right in 2016
plot_finish_gender_distribution(bm_2015_results,2015)
plot_finish_gender_distribution(bm_2016_results,2016)

plot_finish_by_bib <- function(boston_data,race_year){
  #' This function will return a plot object that will visualize the relationship between Bib number and finish time
  #' The input is a Boston Marathon data frame, the data must be pre-processed to get wave and redact missing values
  
  boston_data %>%
    mutate(M.F = ifelse(M.F == "F","Female","Male")) %>%
    group_by(M.F) %>%
    ggplot() + geom_point(aes(x=Bib,y=Final_Time,color=as.factor(Wave))) +
    guides(color=guide_legend(title="Wave")) +
    facet_grid(cols = vars(M.F)) +
    xlab("Bib Number") + ylab("Final Time (minutes)") +
    ggtitle(paste("Boston Non-Elite Bib Race Performance",race_year)) +
    ylim(c(0,500)) + scale_color_igv()
}
## There is clearly a trend with Bib number.
## It is interesting that the variance appears higher for men than for women at lower bib numbers
plot_finish_by_bib(bm_2015_results,2015)
plot_finish_by_bib(bm_2016_results,2016)

plot_gender_split_by_bib <- function(boston_data,race_year){
  #' This function will return a plot object that will visualize how the gender split changes through the bib-numbers
  #' The input is a Boston Marathon data frame, the data must be pre-processed to get wave and redact missing values
  
  boston_data %>%
    mutate(IsFemale = ifelse(M.F == "F",1,0)) %>%
    ggplot(aes(x=Bib,y=IsFemale,color=as.factor(Wave))) + geom_smooth() +
    guides(color = guide_legend(title="Wave")) +
    xlab("Bib Number") + ylab("Proportion Female") +
    ggtitle(paste("Gender Split by Bib Number and Wave |",race_year)) +
    ylim(c(0,1)) + scale_color_igv()
}
## In both years the fraction of women in each cluster of bib numbers increases from the low bibs through the end of wave 2
## Wave 3 is predominantly women. Wave 4 starts mostly women and then trends back down towards majority male toward the end of the wave
## There may be some Yule-Simpson type effects that could impact linear model performance that we will need to be careful of
plot_gender_split_by_bib(bm_2015_results,2015)
plot_gender_split_by_bib(bm_2016_results,2016)


plot_finish_by_Half_pace <- function(boston_data,race_year){
  #' This function will return a plot object that will visualize the linear relationship between race time at the half and finish time
  #' The input is a Boston Marathon data frame, the data must be pre-processed to get wave and redact missing values
  
  boston_data %>%
    mutate(M.F = ifelse(M.F == "F","Female","Male")) %>%
    group_by(M.F) %>%
    ggplot() + geom_point(aes(x=PaceAtHalf,y=Final_Time,color=as.factor(Wave))) +
    guides(color=guide_legend(title="Wave")) +
    facet_grid(cols = vars(M.F)) +
    xlab("Pace at Half (minutes/mile)") + ylab("Final Time (minutes)") +
    ggtitle(paste("Boston Non-Elite Bib Race Performance - Finish vs. Start",race_year)) +
    ylim(c(100,500)) + xlim(c(5,20)) + scale_color_igv()
}
## It is unsurprizing that faster paces at 5km correlate with faster finish times
## However there is clearly a lot of variance that linear models will have a hard time fitting
plot_finish_by_Half_pace(bm_2015_results,2015)
plot_finish_by_Half_pace(bm_2016_results,2016)

plot_neg_pos_split <- function(boston_data,race_year){
  #' This function will create a plot object will visualize the distribution of how much runners slowed down in the second half of the race
  #' The input is a Boston Marathon data frame, the data must be pre-processed to get wave and redact missing values
  
  boston_data %>%
    mutate(M.F = ifelse(M.F == "F","Female","Male")) %>%
    mutate(SecondHalfSplit = (Final_Time - Half_Time) / Half_Time) %>%
    group_by(M.F) %>%
    ggplot() + geom_histogram(aes(x=SecondHalfSplit,fill=as.factor(Wave)),color="black",binwidth = 0.01) +
    guides(fill=guide_legend(title="Wave")) +
    facet_grid(cols = vars(M.F)) +
    xlab("Ratio of Second Half Time to First Half Time") +
    ggtitle(paste("Boston Non-Elite Bib Race Second Half Split",race_year)) +
    xlim(c(0.9,1.5)) + scale_fill_igv()
}
## Men appear to slow down more in general in the second half of the race than women
## Split by gender, the later waves appear to have a broader distribution of how much the runners slow down in the second half
plot_neg_pos_split(bm_2015_results,2015)
plot_neg_pos_split(bm_2016_results,2016)

bib_age_surface <- function(boston_data,race_year,include_wave4){
  #' This function will generate a surface plot of Finish Time vs. Age and Bib number
  #' The input is a pre-processed Boston Marathon dataframe, the race year for the title and a TRUE/FALSE flag
  #' If the TRUE/FALSE flag is flase the wave 4 data will not be flagged
  
  y = boston_data$Age
  x = boston_data$Bib
  z = boston_data$Final_Time
  wave = as.numeric(boston_data$Wave)
  df = data.frame(cbind(x,y,z,wave))
  
  if (include_wave4){
    x = df$x
    y = df$y
    z = df$z
  } else {
    df = subset(df,df$wave < 4)
    x = df$x
    y = df$y
    z = df$z
  }
  
  # Compute the linear regression (z = ax + by + d)
  grid.lines = 25
  fit <- lm(z ~ poly(x,1) + poly(y,1))
  # predict values on regular xy grid
  x.pred <- seq(min(x), max(x), length.out = grid.lines)
  y.pred <- seq(min(y), max(y), length.out = grid.lines)
  xy <- expand.grid( x = x.pred, y = y.pred)
  z.pred <- matrix(predict(fit, newdata = xy), 
                   nrow = grid.lines, ncol = grid.lines)
  fitpoints <- predict(fit)
  
  scatter3D(x, y, z,
            bty="g", phi = 10, alpha = 0.2,
            clab = c("Finish Time","(min)"),
            ticktype = "detailed",
            ylab = "Age (years)",
            xlab = "Bib", zlab = "Finish Time (min)",
            surf = list(x = x.pred, y = y.pred, z = z.pred,
                        facets = NA, fit = fitpoints),
            main = paste(c("Finish Time by Bib and Age, Boston ",race_year)))
}
## These take a while to render, remove the comment symbol to see each plot
## Surface plots are commented out for faster script speed when not executing line-by-line
## Bib appears to dominate over Age in terms of the shape of the fitting surface but variance is quite large
## There may be some clustering of the data which could be interesting
# bib_age_surface(bm_2015_results,2015,T)
# bib_age_surface(bm_2015_results,2015,F)
# bib_age_surface(bm_2016_results,2016,T)
# bib_age_surface(bm_2016_results,2016,F)

bib_pace_surface <- function(boston_data,race_year){
  #' This function will generate a surface plot of Finish/Half Time vs. Bib and Half/10k Pace number
  #' The input is a pre-processed Boston Marathon dataframe, the race year for the title
  
  y = boston_data$Age
  x = boston_data$NrmPace_Half
  z = boston_data$Final_Time / boston_data$Half_Time
  df = data.frame(cbind(x,y,z))
  df = subset(df,df$x < 1.2)
  df = subset(df,df$z < 3)
  x = df$x
  y = df$y
  z = df$z
  
  # Compute the linear regression (z = a(x^2) + b(y^2) + d)
  grid.lines = 25
  fit <- lm(z ~ poly(x,2) + poly(y,2))
  # predict values on regular xy grid
  x.pred <- seq(min(x), max(x), length.out = grid.lines)
  y.pred <- seq(min(y), max(y), length.out = grid.lines)
  xy <- expand.grid( x = x.pred, y = y.pred)
  z.pred <- matrix(predict(fit, newdata = xy), 
                   nrow = grid.lines, ncol = grid.lines)
  fitpoints <- predict(fit)
  
  scatter3D(x, y, z,
            bty="g", phi = 10, alpha = 0.2,
            clab = c("Finish Time","(min)"),
            ticktype = "detailed",
            ylab = "Age (years)",
            xlab = "Half/10k Pace", zlab = "Finish/Half Time",
            surf = list(x = x.pred, y = y.pred, z = z.pred,
                        facets = NA, fit = fitpoints),
            main = paste("Slowdown vs Age, Boston ",race_year))
}
## These take a while to render, remove the comment symbol to see each plot
## Surface plots are commented out for faster script speed when not executing line-by-line
## The ratio of finish time to half time appears super-linear vs. the ratio of half time / 10km time
## Age does not appear to significantly modify this fit
# bib_pace_surface(bm_2015_results,2015)
# bib_pace_surface(bm_2016_results,2016)

bib_time_surface <- function(boston_data,race_year,include_wave4){
  #' This function will generate a surface plot of Finish/Half Time vs. Bib and Half/10k Pace number
  #' The input is a pre-processed Boston Marathon dataframe, the race year for the title and a TRUE/FALSE flag
  #' If the TRUE/FALSE flag is flase the wave 4 data will not be included in the plot
 
  y = boston_data$Bib
  x = boston_data$Half_Time
  z = boston_data$Final_Time
  wave = as.numeric(boston_data$Wave)
  df = data.frame(cbind(x,y,z,wave))
  
  if (include_wave4){
    x = df$x
    y = df$y
    z = df$z
  } else {
    df = subset(df,df$wave < 4)
    x = df$x
    y = df$y
    z = df$z
  }
  
  # Compute the linear regression (z = a(x^2) + b(y^2) + d)
  grid.lines = 25
  fit <- lm(z ~ poly(x,2) + poly(y,2))
  # predict values on regular xy grid
  x.pred <- seq(min(x), max(x), length.out = grid.lines)
  y.pred <- seq(min(y), max(y), length.out = grid.lines)
  xy <- expand.grid( x = x.pred, y = y.pred)
  z.pred <- matrix(predict(fit, newdata = xy), 
                   nrow = grid.lines, ncol = grid.lines)
  fitpoints <- predict(fit)
  
  scatter3D(x, y, z,
            bty="g", phi = 10, alpha = 0.2,
            clab = c("Finish Time","(min)"),
            ticktype = "detailed",
            ylab = "Bib",
            xlab = "Half Marathon Time (min)", zlab = "Finish Time (min)",
            surf = list(x = x.pred, y = y.pred, z = z.pred,
                        facets = NA, fit = fitpoints),
            main = paste("Finish vs. Bib and Half Time, Boston ",race_year))
}
## These take a while to render, remove the comment symbol to see each plot
## Surface plots are commented out for faster script speed when not executing line-by-line
## The runner's time at the half marathon appears to contain most of the Bib number information
## Bib number and half marathon time are correlated since runners are assigned their number based on qualification time
# bib_time_surface(bm_2015_results,2015,T)
# bib_time_surface(bm_2015_results,2015,F)
# bib_time_surface(bm_2016_results,2016,T)
# bib_time_surface(bm_2016_results,2016,F)


bib_5ktime_surface <- function(boston_data,race_year,include_wave4){
  #' This function will generate a surface plot of Finish/Half Time vs. Bib and Half/10k Pace number
  #' The input is a pre-processed Boston Marathon dataframe, the race year for the title and a TRUE/FALSE flag
  #' If the TRUE/FALSE flag is flase the wave 4 data will not be included in the plot

  y = boston_data$Bib
  x = boston_data$KM05_Time
  z = boston_data$Final_Time
  wave = as.numeric(boston_data$Wave)
  df = data.frame(cbind(x,y,z,wave))

    if (include_wave4){
    x = df$x
    y = df$y
    z = df$z
  } else {
    df = subset(df,df$wave < 4)
    x = df$x
    y = df$y
    z = df$z
  }
  
  # Compute the linear regression (z = a(x^2) + b(y^2) + d)
  grid.lines = 25
  fit <- lm(z ~ poly(x,2) + poly(y,2))
  # predict values on regular xy grid
  x.pred <- seq(min(x), max(x), length.out = grid.lines)
  y.pred <- seq(min(y), max(y), length.out = grid.lines)
  xy <- expand.grid( x = x.pred, y = y.pred)
  z.pred <- matrix(predict(fit, newdata = xy), 
                   nrow = grid.lines, ncol = grid.lines)
  fitpoints <- predict(fit)
  
  scatter3D(x, y, z,
            bty="g", phi = 10, alpha = 0.2,
            clab = c("Finish Time","(min)"),
            ticktype = "detailed",
            ylab = "Bib",
            xlab = "5km Time (min)", zlab = "Finish Time (min)",
            surf = list(x = x.pred, y = y.pred, z = z.pred,
                        facets = NA, fit = fitpoints),
            main = paste("Finish vs. Bib and 5km Time, Boston ",race_year))
}
## These take a while to render, remove the comment symbol to see each plot
## Surface plots are commented out for faster script speed when not executing line-by-line
## Bib number looks like it may add a bit more curvature than it did at the half marathon but still less so than the runner's time
# bib_5ktime_surface(bm_2015_results,2015,T)
# bib_5ktime_surface(bm_2015_results,2015,F)
# bib_5ktime_surface(bm_2016_results,2016,T)
 bib_5ktime_surface(bm_2016_results,2016,F)

plot_splis_dist_5k_addative <- function(boston_data,race_year){
  #' This function will return a plot of the distribution of runner slow-downs from the 5km point
  #' The slow down is in the units of minutes for this plot
  #' The required input is a Boston Marathon results dataframe and the race year
  
  boston_data %>%
    mutate(naive_residuals_5k = Final_Time - naive_KM05_Finish,
           M.F = ifelse(M.F == "F","Female","Male")) %>%
    ggplot() + geom_histogram(aes(x=naive_residuals_5k,fill=M.F),color="black",binwidth=2.5) +
    guides(fill=guide_legend(title="Gender")) + xlab("Final Time - est Final Time from 5km") +
    ylab("Count") + ggtitle(paste("Distribution of Slow Down from 5KM",race_year,"Boston Marathon")) +
    xlim(c(-50,150)) + scale_fill_igv()
}
## The degree to which runners slowed down from their 5km pace is much broader in 2016 than 2015
plot_splis_dist_5k_addative(bm_2015_results,2016)
plot_splis_dist_5k_addative(bm_2016_results,2016)

plot_splis_dist_5k_multiplicative <- function(boston_data,race_year,bw){
  #' This function will return a plot of the distribution of runner slow-downs from the 5km point
  #' The slow down is unitless, it is the ratio of final pace to 5km pace
  boston_data %>% mutate(naive_multiplier_5k = (Final_Time / 26.2188) / (KM05_Time / 3.10686),
                         M.F = ifelse(M.F == "F","Female","Male")) %>%
    ggplot() + geom_histogram(aes(x=naive_multiplier_5k,fill=M.F),color="black",binwidth=bw) +
    guides(fill=guide_legend(title="Gender")) + xlab("Ratio: Final Pace / Pace at 5KM") +
    ylab("Count") + ggtitle(paste("Distribution of Slow Down from 5KM",race_year,"Boston Marathon")) +
    xlim(c(0.75,1.75)) + scale_fill_igv()
}
## The same effect is seen when looking at ratios rather than addative residuals
## The degree to which runners slowed down from their 5km pace is much broader in 2016 than 2015
plot_splis_dist_5k_multiplicative(bm_2015_results,2015,bw=0.01)
plot_splis_dist_5k_multiplicative(bm_2016_results,2016,bw=0.01)

plot_resids_by_age_additive <- function(dist,boston_data,race_year,naive){
  #' This function visualizes the additive residuals to the naive 15km prediction by age, gender and wave
  #' Function input is the distance of the split, the data-set, the race year and the naive predictions
  #' Output is a plot of the residuals to the naive (B.A.A.) prediction by age, split by sex

  boston_data %>%
    mutate(naive_residuals = Final_Time - naive, M.F = ifelse(M.F == "F","Female","Male")) %>%
    group_by(M.F) %>%
    ggplot() + geom_point(alpha = 0.05, aes(x=as.numeric(Age),y=naive_residuals,color=as.factor(Wave))) +
    geom_smooth(se = F,span = 0.2,aes(x=as.numeric(Age),y=naive_residuals,color=as.factor(Wave))) +
    xlab("Runner Age (years)") + ylab(paste("Residual at",dist,"(minutes)")) +
    ggtitle(paste("Residuals to Naieve Prediction at", dist,"by Age and Gender")) + 
    guides(color=guide_legend(title="Wave")) +
    facet_grid(cols = vars(M.F)) +
    ylim(c(-100,100))  + scale_color_igv()
}
## We can see the residual is biased generally positive throughout the race
## There are some trends with age but wave and gender appear stronger predictors
## As noted in the surface plot analysis, Age information appears mostly contained in speed and bib features
## Remove the comment sign to render these plots
# plot_resids_by_age_additive("5k",bm_2015_results,2015,bm_2015_results$naive_KM05_Finish)
# plot_resids_by_age_additive("5k",bm_2016_results,2016,bm_2016_results$naive_KM05_Finish)
# plot_resids_by_age_additive("10k",bm_2015_results,2015,bm_2015_results$naive_KM10_Finish)
# plot_resids_by_age_additive("10k",bm_2016_results,2016,bm_2016_results$naive_KM10_Finish)
# plot_resids_by_age_additive("15k",bm_2015_results,2015,bm_2015_results$naive_KM15_Finish)
# plot_resids_by_age_additive("15k",bm_2016_results,2016,bm_2016_results$naive_KM15_Finish)
# plot_resids_by_age_additive("20k",bm_2015_results,2015,bm_2015_results$naive_KM20_Finish)
# plot_resids_by_age_additive("20k",bm_2016_results,2016,bm_2016_results$naive_KM20_Finish)
# plot_resids_by_age_additive("25k",bm_2015_results,2015,bm_2015_results$naive_KM25_Finish)
# plot_resids_by_age_additive("25k",bm_2016_results,2016,bm_2016_results$naive_KM25_Finish)
# plot_resids_by_age_additive("30k",bm_2015_results,2015,bm_2015_results$naive_KM30_Finish)
# plot_resids_by_age_additive("30k",bm_2016_results,2016,bm_2016_results$naive_KM30_Finish)
# plot_resids_by_age_additive("35k",bm_2015_results,2015,bm_2015_results$naive_KM35_Finish)
# plot_resids_by_age_additive("35k",bm_2016_results,2016,bm_2016_results$naive_KM35_Finish)
# plot_resids_by_age_additive("40k",bm_2015_results,2015,bm_2015_results$naive_KM40_Finish)
# plot_resids_by_age_additive("40k",bm_2016_results,2016,bm_2016_results$naive_KM40_Finish)

plot_resids_by_age_multiplicative <- function(dist,boston_data,race_year,pace){
  #' This function visualizes the additive residuals to the naive 15km prediction by age, gender and wave
  #' Function input is the distance of the split, the data-set, the race year and the pace up to the split
  #' Output is a plot of the residuals to the naive (B.A.A.) prediction by age, split by sex

  boston_data %>%
    mutate(naive_multiplier = (Final_Time / (26.2188 * pace)), M.F = ifelse(M.F == "F","Female","Male")) %>%
    group_by(M.F) %>%
    ggplot() + geom_point(alpha = 0.05, aes(x=as.numeric(Age),y=naive_multiplier,color=as.factor(Wave))) +
    geom_smooth(se = F,span = 0.2,aes(x=as.numeric(Age),y=naive_multiplier,color=as.factor(Wave))) +
    xlab("Runner Age (years)") + ylab(paste("Finish Pace Ratio at",dist,"(unitless)")) +
    ggtitle(paste("Final Pace Ratio:", dist,",",race_year)) + 
    guides(color=guide_legend(title="Wave")) +
    facet_grid(cols = vars(M.F)) +
    ylim(c(0.9,1.25)) + scale_color_igv()
}
## We can see the ratio is biased generally higher than 1
## There are some trends with age
## Since these trends are often either flat or non-linear Age looks like a better fit for the regression tree models than the linear models
## As noted in the surface plot analysis, Age information appears mostly contained in speed and bib features
## Remove the comment sign to render these plots
# plot_resids_by_age_multiplicative("5k",bm_2015_results,2015,bm_2015_results$PaceAt5K)
# plot_resids_by_age_multiplicative("5k",bm_2016_results,2016,bm_2016_results$PaceAt5K)
# plot_resids_by_age_multiplicative("10k",bm_2015_results,2015,bm_2015_results$PaceAt10K)
# plot_resids_by_age_multiplicative("10k",bm_2016_results,2016,bm_2016_results$PaceAt10K)
# plot_resids_by_age_multiplicative("15k",bm_2015_results,2015,bm_2015_results$PaceAt15K)
# plot_resids_by_age_multiplicative("15k",bm_2016_results,2016,bm_2016_results$PaceAt15K)
# plot_resids_by_age_multiplicative("20k",bm_2015_results,2015,bm_2015_results$PaceAt20K)
# plot_resids_by_age_multiplicative("20k",bm_2016_results,2016,bm_2016_results$PaceAt20K)
# plot_resids_by_age_multiplicative("25k",bm_2015_results,2015,bm_2015_results$PaceAt25K)
# plot_resids_by_age_multiplicative("25k",bm_2016_results,2016,bm_2016_results$PaceAt25K)
# plot_resids_by_age_multiplicative("30k",bm_2015_results,2015,bm_2015_results$PaceAt30K)
# plot_resids_by_age_multiplicative("30k",bm_2016_results,2016,bm_2016_results$PaceAt30K)
# plot_resids_by_age_multiplicative("35k",bm_2015_results,2015,bm_2015_results$PaceAt35K)
# plot_resids_by_age_multiplicative("35k",bm_2016_results,2016,bm_2016_results$PaceAt35K)
# plot_resids_by_age_multiplicative("40k",bm_2015_results,2015,bm_2015_results$PaceAt40K)
# plot_resids_by_age_multiplicative("40k",bm_2016_results,2016,bm_2016_results$PaceAt40K)

plot_rf_imp <- function(model,title){
  #' This function plots the variable importance from a random forest model
  #' The input is the model and the plot title
  #' The function will create the plot object
  
  rf_mod_imp = varImp(model)
  plot(rf_mod_imp, main = title,
       xlab="Feature Importance")
}


#########################################################
##
## Model Prediction Functions
## This block will define functions which make final time predictions at each split
##
#########################################################

predict_rf_05km <- function(boston_data){
  #' This function predicts the final time at 5km
  #' The input is a pre-processed Boston Marathon Results data frame
  #' The prediction is the average of the linear model + a random forest residual and the linear model times a random forest ratio
  #' The function will return a vector of predicted values
   
  ## Bias is reduced with a linear model
  lin_preds = predict(mod_lm_05,newdata = boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_05k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_05k_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_10km <- function(boston_data){
  #' This function predicts the final time at 10km
  #' The input is a pre-processed Boston Marathon Results data frame
  #' The prediction is the average of the linear model + a random forest residual and the linear model times a random forest ratio
  #' The function will return a vector of predicted values
  
  ## Bias is reduced with a linear model
  lin_preds = predict(mod_lm_10,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_10k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_10k_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_15km <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at 15km

  ## Bias is reduced with a linear model
  lin_preds = predict(mod_lm_15,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_15k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_15k_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_20km <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at 20km
  
  ## Bias is reduced with a linear model  
  lin_preds = predict(mod_lm_20,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_20k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_20k_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_half <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at the half marathon

  ## Bias is reduced with a linear model 
  lin_preds = predict(mod_lm_Half,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_half_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_half_mult,newdata = boston_data)
  
  # Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_25km <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at 25km
  
  ## Bias is reduced with a linear model 
  lin_preds = predict(mod_lm_25,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_25k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_25_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_30km <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at 30km
  
  ## Bias is reduced with a linear model 
  lin_preds = predict(mod_lm_30,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_30k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_30_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_35km <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at 35km
  
  ## Bias is reduced with a linear model
  lin_preds = predict(mod_lm_35,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_35k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_35_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

predict_rf_40km <- function(boston_data){
  #' See the docstring from predict_rf10km() the only difference is that this predicts at 40km
  
  ## Bias is reduced with a linear model
  lin_preds = predict(mod_lm_40,boston_data)
  
  ## Variance is modeled with two different random forest models
  add_resids = predict(mod_rf_40k_add,newdata = boston_data)
  mult_ratio = predict(mod_rf_40_mult,newdata = boston_data)
  
  ## Each random forest model adjusts the linear model prediction and the average of the two models is returned
  preds = ((lin_preds + add_resids) + (lin_preds * mult_ratio)) / 2
  return(preds)
}

#########################################################
##
## Create Plot Objects To Insert in Report
## These are some plots that I'm considering most strongly for the report
## Plots are assigned to objects to simplify the chunks at their insertion point in the markdown file
##
#########################################################

## This is a general finish time visualization
gen_dist_2015_2016 <- ggarrange(plot_finish_gender_distribution(bm_2015_results,"2015") + xlab(""),
                                plot_finish_gender_distribution(bm_2016_results,"2016"),ncol = 1, nrow = 2)


## 2016 had a more general trend toward slowing down in the second half
dist_add_slow_2015_2016 <- ggarrange(plot_splis_dist_5k_addative(bm_2015_results,2015)+xlab(""),
                                     plot_splis_dist_5k_addative(bm_2016_results,2016),
                                     ncol = 1, nrow = 2)

## 2016 had a more general trend toward slowing down in the second half
dist_mult_slow_2015_2016 <- ggarrange(plot_splis_dist_5k_multiplicative(bm_2015_results,2015,0.01)+xlab(""),
                                      plot_splis_dist_5k_multiplicative(bm_2016_results,2016,0.01),
                                      ncol = 1, nrow = 2)

## Each wave appears to slow down a bit more than the previous with increasing variation
dist_add_slow_half <- ggarrange(plot_neg_pos_split(bm_2015_results,2015) + xlab(""),
                                plot_neg_pos_split(bm_2016_results,2016),ncol = 1, nrow = 2)

## There appear to be general trends in slow down which differ by wave at the 5k
scat_mult_slow_wave_gen_5k <- ggarrange(plot_resids_by_age_multiplicative("5k",bm_2015_results,
                                                                          2015,bm_2015_results$PaceAt5K) +
                                          theme(legend.position = "none"),
                                        plot_resids_by_age_multiplicative("5k",bm_2016_results,
                                                                          2016,bm_2016_results$PaceAt5K) + ylab(""),
                                        ncol = 2, nrow = 1)

## The pace ratio appears to reverse by the half...this may suggest that older runners may settle into their final pace better than younger?
scat_mult_slow_wave_gen_half <- ggarrange(plot_resids_by_age_multiplicative("Half",bm_2015_results,
                                                                            2015,bm_2015_results$PaceAtHalf) +
                                            theme(legend.position = "none"),
                                          plot_resids_by_age_multiplicative("Half",bm_2016_results,
                                                                            2016,bm_2016_results$PaceAtHalf) + ylab(""),
                                          ncol = 2, nrow = 1)


## Get the RMSE performance from the baseline mode on each of the 2015 and 2016 data sets
bm2015_naive <- get_naive_performance(bm_2015_results)
bm2016_naive <- get_naive_performance(bm_2016_results)

## Create a plot object the naive mode RMSE at each split for each year
naive_perf_2015_2016 <- ggplot() + geom_point(aes(x=Distance,y=Naive.RMSE),data = bm2015_naive,color="blue") +
  geom_line(aes(x=Distance,y=Naive.RMSE),data = bm2015_naive,color="blue") +
  geom_point(aes(x=Distance,y=Naive.RMSE),data = bm2016_naive,color="red") +
  geom_line(aes(x=Distance,y=Naive.RMSE),data = bm2016_naive,color="red") +
  xlab("Race Distance Completed (km)") + ylab("Naive Model RMSE (minutes)") + 
  ggtitle("Baseline BAA Projected Time Performance | 2015 and 2016") +
  annotate("text",x=30,y=29,label = "2015 Boston Marathon",color="blue") +
  annotate("text",x=30,y=27.5,label = "2016 Boston Marathon",color="red") +
  xlim(c(0,42)) + ylim(c(0,30)) + 
  geom_rect(aes(xmin = 18,xmax = 27, ymin = 11, ymax = 23),
            color = "black",fill = "light green", alpha=0.2) +
  annotate("text",x=22.5,y=24.5,
           label = "Key Prediction Splits",color="black")

## This visualization just grounds what I'm shooting at with this project
## This highlights how significant year-to-year differences can be to the B.A.A. baseline prediction of constant pace
## This plot isn't particularly relevent to the report so the block that generates it will be left out of the Markdown file
naive_perf_2015_2016


#########################################################
##
## Create Training and Testing Sets
## Models will be built with a training set and tuned to give the best performance on a random test set
## Since 2015 and 2016 are characteristically different we will mix both into training and testing sets
## The expectation is that the linear models will somewhat split the difference between 2015 and 2016
## The random forest models will be expected to learn from the data what are the signs of slowdown to predict either
## The race year will be withheld from all models since we will need to predict 2017 with no 2017 information
##
#########################################################

## Make testing and training data from 2015 and 2016 data
## 67% training and 33% testing is selected to mirror the 2-races for development (2015 + 2016) and 1-race for validation (2017) split

## Make training and testing sets from 2015 data
test_train_list <- make_train_test_sets(bm_2015_results,0.33,42)
training <- test_train_list[[1]]
testing <- test_train_list[[2]]

## Append on training and testing data from 2016
test_train_list <- make_train_test_sets(bm_2016_results,0.33,42)
training <- rbind(training,test_train_list[[1]])
testing <- rbind(testing,test_train_list[[2]])

## The after tuning on the training set to optimize results on the testing set we will retrain on the combined data set
## This just recombines the two partitions into that final training data set
final_training <- rbind(training,testing)


#########################################################
##
## Bias Reduction: Linear Models
## The linear models will be very simple.
## Their job is to reduce the bias so that our more complex models can focus primarily on the variance.
## Since we are not tuning these basic models we will jump straight to training on the final_training dataset
##
#########################################################

## Spot-check feature significance and Est OOS RMSE
## A RMSE ratio < 1 indicates improvement over the baseline
## The spot check is not necessary for the report so it is not included in the Markdown script
## Un-comment out the code below to see the model summary and ratio of the model RMSE to baseline RMSE on the test set

## 5km Linear Model spot check
# mod_lm_05_a <- train(Final_Time ~ KM05_Time + Wave + M.F,data = training,method = "lm")
# summary(mod_lm_05_a)
# RMSE(testing$Final_Time,predict(mod_lm_05_a,newdata = testing)) / (RMSE(testing$Final_Time,testing$naive_KM05_Finish))

## 10km Linear Model spot check
# mod_lm_10_a <- train(Final_Time ~ KM10_Time + Wave + M.F,data = training,method = "lm")
# summary(mod_lm_10_a)
# RMSE(testing$Final_Time,predict(mod_lm_10_a,newdata = testing)) / (RMSE(testing$Final_Time,testing$naive_KM10_Finish))

## 20km Linear Model spot check
# mod_lm_20_a <- train(Final_Time ~ KM20_Time + Wave + M.F,data = training,method = "lm")
# summary(mod_lm_20_a)
# RMSE(testing$Final_Time,predict(mod_lm_20_a,newdata = testing)) / (RMSE(testing$Final_Time,testing$naive_KM20_Finish))

## 30km Linear Model spot check
# mod_lm_30_a <- train(Final_Time ~ KM30_Time + Wave + M.F,data = training,method = "lm")
# summary(mod_lm_30_a)
# RMSE(testing$Final_Time,predict(mod_lm_30_a,newdata = testing)) / (RMSE(testing$Final_Time,testing$naive_KM30_Finish))

## 30km Linear Model spot check
# mod_lm_40_a <- train(Final_Time ~ KM40_Time + Wave + M.F,data = training,method = "lm")
# summary(mod_lm_40_a)
# RMSE(testing$Final_Time,predict(mod_lm_40_a,newdata = testing)) / (RMSE(testing$Final_Time,testing$naive_KM40_Finish))

## Linear Prediction Model Training -- Final Time Predictions
## The Wave and gender are selected for categorical inclusion with the raw time based on the EDA
mod_lm_05 <- train(Final_Time ~ KM05_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_10 <- train(Final_Time ~ KM10_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_15 <- train(Final_Time ~ KM15_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_20 <- train(Final_Time ~ KM20_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_Half <- train(Final_Time ~ Half_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_25 <- train(Final_Time ~ KM25_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_30 <- train(Final_Time ~ KM30_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_35 <- train(Final_Time ~ KM35_Time + Wave + M.F,data = final_training,method = "lm")
mod_lm_40 <- train(Final_Time ~ KM40_Time + Wave + M.F,data = final_training,method = "lm")

## Linear Prediction Model Training -- Next Check-Point Time Predictions
## At each check-point we will also predict the time the runner will reach the next check-point
## Once the runner arrives at the next check point their residual to the prediction can be used as a feature in the Random Forest models
pred10k_lm <- train(KM10_Time ~ KM05_Time + Wave + M.F,data = final_training,method = "lm")
pred15k_lm <- train(KM15_Time ~ KM10_Time + Wave + M.F,data = final_training,method = "lm")
pred20k_lm <- train(KM20_Time ~ KM15_Time + Wave + M.F,data = final_training,method = "lm")
pred_half_lm <- train(Half_Time ~ KM20_Time + Wave + M.F,data = final_training,method = "lm")
pred25k_lm <- train(KM25_Time ~ Half_Time + Wave + M.F,data = final_training,method = "lm")
pred30k_lm <- train(KM30_Time ~ KM25_Time + Wave + M.F,data = final_training,method = "lm")
pred35k_lm <- train(KM35_Time ~ KM30_Time + Wave + M.F,data = final_training,method = "lm")
pred40k_lm <- train(KM40_Time ~ KM35_Time + Wave + M.F,data = final_training,method = "lm")

training <- add_mid_race_miss(training)
testing <- add_mid_race_miss(testing)
final_training <- add_mid_race_miss(final_training)


#########################################################
##
## Variance Reduction: Regression Tree Random Forest Models
## We will be using regression tree random forest models to predict the residuals to or ratio to the linear fit models
## The Regression Tree model is selected due to the high degree of variance seen in the visualizations and the posibility of data clusters
## The Random Forest is selected to use randomization to target better out-of-sample performance
##
#########################################################

## One class of regression tree random forest will attempt to predict addative residuals to the linear model predictions
## To train on these we will predict the final time for each runner in the training set and calculate the difference between the actual and predicted times
training$lm_resids05 <- training$Final_Time - predict(mod_lm_05,training)
training$lm_resids10 <- training$Final_Time - predict(mod_lm_10,training)
training$lm_resids15 <- training$Final_Time - predict(mod_lm_15,training)
training$lm_resids20 <- training$Final_Time - predict(mod_lm_20,training)
training$lm_residsHalf <- training$Final_Time - predict(mod_lm_Half,training)
training$lm_resids25 <- training$Final_Time - predict(mod_lm_25,training)
training$lm_resids30 <- training$Final_Time - predict(mod_lm_30,training)
training$lm_resids35 <- training$Final_Time - predict(mod_lm_35,training)
training$lm_resids40 <- training$Final_Time - predict(mod_lm_40,training)

## The second class of regression tree random forest will attempt to predict the ratio of the final time to the time predicted from the linear model
## To train on these we will predict the final time for each runner in the training set and divide the actual final time by this prediction
training$lm_rato_05 <- training$Final_Time / predict(mod_lm_05,training)
training$lm_rato_10 <- training$Final_Time / predict(mod_lm_10,training)
training$lm_rato_15 <- training$Final_Time / predict(mod_lm_15,training)
training$lm_rato_20 <- training$Final_Time / predict(mod_lm_20,training)
training$lm_rato_Half <- training$Final_Time / predict(mod_lm_Half,training)
training$lm_rato_25 <- training$Final_Time / predict(mod_lm_25,training)
training$lm_rato_30 <- training$Final_Time / predict(mod_lm_30,training)
training$lm_rato_35 <- training$Final_Time / predict(mod_lm_35,training)
training$lm_rato_40 <- training$Final_Time / predict(mod_lm_40,training)


################################
##
## 5km: Random Forest
## We expect the lest benefit at 5km because we have the least new data that the linear model cannot fit
## The widest node range is used at 5km since we do not have a previous model to inform the decision
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size...
  nodes <- c(100,200,500,750,1000,1250,1500,1750,2000)
  setup_for_rf_tune_for()
  for (ns in nodes){
    ## We'll loop through all of the nodes in our grid, train on the training data and record the testing rmse
    ## This model predicts the delta between actual and predicted from the linear model so the prediction is the sum of RF and LM models
    ## This model is intended to capture slowdown or step function changes (injuries, bathroom breaks etc...)
    rf_05k = train(lm_resids05 ~ PaceAt5K + M.F + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_05k,newdata = testing) + predict(mod_lm_05,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    ## This loop will try increasing the random forest size; if our RMSE increases significantly we will try again until the improvement is <1%
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_05k = train(lm_resids05 ~ PaceAt5K + M.F + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_05k,newdata = testing) + predict(mod_lm_05,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one iteration
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 5km random forest addition model on the Final Training data with the optimized ns and nt
  final_training$lm_resids05 <- final_training$Final_Time - predict(mod_lm_05,newdata = final_training)
  mod_rf_05k_add = train(lm_resids05 ~ PaceAt5K + M.F + Age + Bib,data=final_training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    ## We'll loop through all of the nodes in our grid, train on the training data and record the testing rmse
    ## This model predicts the ratio of the actual and predicted from the linear model so the prediction is the product of the two models
    ## This model is intended to capture % slow down by the runner
    rf_05k = train(lm_rato_05 ~ PaceAt5K + M.F + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_05k,newdata = testing) * predict(mod_lm_05,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    ## This loop will try increasing the random forest size; if our RMSE increases significantly we will try again until the improvement is <1%
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_05k = train(lm_rato_05 ~ PaceAt5K + M.F + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_05k,newdata = testing) * predict(mod_lm_05,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 5km random forest multiplication model on the Final Training data
  final_training$lm_rato_05 <- final_training$Final_Time / predict(mod_lm_05,newdata = final_training)
  mod_rf_05k_mult = train(lm_rato_05 ~ PaceAt5K + M.F + Age + Bib,data=final_training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_05k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_05k_add.rds","mod_rf_05k_add.rds")
  mod_rf_05k_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_05k_mult.rds","mod_rf_05k_mult.rds")
}


################################
##
## 10km: Random Forest
## At 10km we have a bit more information so we can start using how much each runner missed their 10km predicted time as a feature
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
## 
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size
  ## As we go toward the center of the race the complexity of the data will increase
  ## Knowing that the node size at 5km was ~600 I'll drop the tests above 1500 to speed up the calculation and add a ns=50 at the start
  nodes <- c(50,100,200,500,750,1000,1250,1500)
  setup_for_rf_tune_for()
  for (ns in nodes){
    ## Same loop as 5km, just a different model
    rf_10k = train(lm_resids10 ~ PaceAt5K + PaceAt10K + M.F + MissAt10k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_10k,newdata = testing) + predict(mod_lm_10,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    ## Still the same as 5km, but with the 10k model
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_10k = train(lm_resids10 ~ PaceAt5K + PaceAt10K + M.F + MissAt10k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_10k,newdata = testing) + predict(mod_lm_10,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one iteration
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 10km additive random forest model on the Final Training data
  final_training$lm_resids10 <- final_training$Final_Time - predict(mod_lm_10,newdata = final_training)
  mod_rf_10k_add = train(lm_resids10 ~ PaceAt5K + PaceAt10K + M.F + MissAt10k_add + Age + Bib,data=final_training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_10k = train(lm_rato_10 ~ PaceAt5K + PaceAt10K + M.F + MissAt10k_mult + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_10k,newdata = testing) * predict(mod_lm_10,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_10k = train(lm_rato_10 ~ PaceAt5K + PaceAt10K + M.F + MissAt10k_mult + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_10k,newdata = testing) * predict(mod_lm_10,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 10km random forest multiplication model on the Final Training data
  final_training$lm_rato_10 <- final_training$Final_Time / predict(mod_lm_10,newdata = final_training)
  mod_rf_10k_mult = train(lm_rato_10 ~ PaceAt5K + PaceAt10K + M.F + MissAt10k_mult + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_10k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_10k_add.rds","mod_rf_10k_add.rds")
  mod_rf_10k_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_10k_mult.rds","mod_rf_10k_mult.rds")
}


################################
##
## 15km: Random Forest
## The 15km model is very similar to the model at 10km, I won't start adding normalized paces until 20km
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At 10km our node size was <400 so we'll just test 50-750 node sizes
  nodes <- c(50,100,200,500,750)
  setup_for_rf_tune_for()
  for (ns in nodes){
    ## Same loop as at 5km and 10km
    rf_15k = train(lm_resids15 ~ PaceAt5K + PaceAt10K + PaceAt15K + M.F + MissAt15k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_15k,newdata = testing) + predict(mod_lm_15,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_15k = train(lm_resids15 ~ PaceAt5K + PaceAt10K + PaceAt15K + M.F + MissAt15k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_15k,newdata = testing) + predict(mod_lm_15,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 15km additive random forest model on the Final Training data
  final_training$lm_resids15 <- final_training$Final_Time - predict(mod_lm_15,newdata = final_training)
  mod_rf_15k_add = train(lm_resids15 ~ PaceAt5K + PaceAt10K + PaceAt15K + M.F + MissAt15k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)

    
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_15k = train(lm_rato_15 ~ PaceAt5K + PaceAt10K + PaceAt15K + M.F + MissAt15k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_15k,newdata = testing) * predict(mod_lm_15,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_15k = train(lm_rato_15 ~ PaceAt5K + PaceAt10K + PaceAt15K + M.F + MissAt15k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_15k,newdata = testing) * predict(mod_lm_15,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}
  
  ## Train the final 15km random forest multiplication model on the Final Training data
  final_training$lm_rato_15 <- final_training$Final_Time / predict(mod_lm_15,newdata = final_training)
  mod_rf_15k_mult = train(lm_rato_15 ~ PaceAt5K + PaceAt10K + PaceAt15K + M.F + MissAt15k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_15k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_15k_add.rds","mod_rf_15k_add.rds")
  mod_rf_15k_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_15k_mult.rds","mod_rf_15k_mult.rds")
}

################################
##
## 20km: Random Forest
## We'll also start including the Normalized Pace (normalized to the 10k time) as a random forest feature
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At 15km our node size was 132 so we'll go from 25 to 500
  nodes <- c(25,50,100,200,500)
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_20k = train(lm_resids20 ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + M.F + MissAt20k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_20k,newdata = testing) + predict(mod_lm_20,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_20k = train(lm_resids20 ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + M.F + MissAt20k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_20k,newdata = testing) + predict(mod_lm_20,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 20km additive random forest model on the Final Training data
  final_training$lm_resids20 <- final_training$Final_Time - predict(mod_lm_20,newdata = final_training)
  mod_rf_20k_add = train(lm_resids20 ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + M.F + MissAt20k_add + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_20k = train(lm_rato_20 ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + M.F + MissAt20k_mult + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_20k,newdata = testing) * predict(mod_lm_20,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_20k = train(lm_rato_20 ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + M.F + MissAt20k_mult + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_20k,newdata = testing) * predict(mod_lm_20,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 20km random forest multiplication model on the Final Training data
  final_training$lm_rato_20 <- final_training$Final_Time / predict(mod_lm_20,newdata = final_training)
  mod_rf_20k_mult = train(lm_rato_20 ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + M.F + MissAt20k_mult + Age + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_20k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_20k_add.rds","mod_rf_20k_add.rds")
  mod_rf_20k_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_20k_mult.rds","mod_rf_20k_mult.rds")
}

ggarrange(plot_rf_imp(mod_rf_20k_mult,"Ratio Model"),plot_rf_imp(mod_rf_20k_add,"Residual Model"),ncol = 2, nrow = 1)
## We'll drop Age and Gender at the Half since they are the least significant terms in the model

################################
##
## Half Marathon: Random Forest
## Per the feature importance analysis at the 20km we will drop Gender and Age from the models moving forward
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At 20km our node size was 111 so I'll drop the 500
  nodes <- c(10,25,50,100,200)
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_half = train(lm_residsHalf ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + PaceAtHalf + NrmPace_Half + MissAt20k_add + MissAt_Half_add + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_half,newdata = testing) + predict(mod_lm_Half,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_half = train(lm_residsHalf ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + PaceAtHalf + NrmPace_20K + NrmPace_Half + MissAt20k_add + MissAt_Half_add + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_half,newdata = testing) + predict(mod_lm_Half,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final Half additive random forest model on the Final Training data
  final_training$lm_residsHalf <- final_training$Final_Time - predict(mod_lm_Half,newdata = final_training)
  mod_rf_half_add = train(lm_residsHalf ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + PaceAtHalf + NrmPace_20K + NrmPace_Half + MissAt20k_add + MissAt_Half_add + Bib,data=final_training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_half = train(lm_rato_Half ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + PaceAtHalf + NrmPace_Half + MissAt20k_mult + MissAt_Half_mult + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_half,newdata = testing) * predict(mod_lm_Half,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_half = train(lm_rato_Half ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + PaceAtHalf + NrmPace_Half + MissAt20k_mult + MissAt_Half_mult + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_half,newdata = testing) * predict(mod_lm_Half,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final half marathon random forest model on the Final Training data
  final_training$lm_rato_Half <- final_training$Final_Time / predict(mod_lm_Half,newdata = final_training)
  mod_rf_half_mult = train(lm_rato_Half ~ PaceAt5K + PaceAt10K + PaceAt15K + PaceAt20K + NrmPace_20K + PaceAtHalf + NrmPace_Half + MissAt20k_mult + MissAt_Half_mult + Bib,data=final_training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_half_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_half_add.rds","mod_rf_half_add.rds")
  mod_rf_half_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_half_mult.rds","mod_rf_half_mult.rds")
}

ggarrange(plot_rf_imp(mod_rf_half_mult,"Ratio Model"),plot_rf_imp(mod_rf_half_add,"Residual Model"),ncol = 2, nrow = 1)
## Pace at 10K, Pace at 15K, PAce at 20K and Pace at half are relatively insignificant in both models so we'll drop them in the 25km model


################################
##
## 25kM: Random Forest
## Per the feature importance at the 20km we will drop Gender and Age from the model
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At the half our node size was 69 so I'll keep the same grid
  nodes <- c(10,25,50,100,200)
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_25k = train(lm_resids25 ~ PaceAt5K + NrmPace_20K + NrmPace_Half + NrmPace_25K + MissAt20k_add + MissAt_Half_add + MissAt25k_add + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_25k,newdata = testing) + predict(mod_lm_25,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_25k = train(lm_resids25 ~ PaceAt5K + NrmPace_20K + NrmPace_Half + NrmPace_25K + MissAt20k_add + MissAt_Half_add + MissAt25k_add + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_25k,newdata = testing) + predict(mod_lm_25,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 25km additive random forest model on the Final Training data
  final_training$lm_resids25 <- final_training$Final_Time - predict(mod_lm_25,newdata = final_training)
  mod_rf_25k_add = train(lm_resids25 ~ PaceAt5K + NrmPace_20K + NrmPace_Half + NrmPace_25K + MissAt20k_add + MissAt_Half_add + MissAt25k_add + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_25k = train(lm_rato_25 ~ PaceAt5K + NrmPace_20K + NrmPace_Half + NrmPace_25K + MissAt20k_mult + MissAt_Half_mult + MissAt25k_mult + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_25k,newdata = testing) * predict(mod_lm_25,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_25k = train(lm_rato_25 ~ PaceAt5K + NrmPace_20K + NrmPace_Half + NrmPace_25K + MissAt20k_mult + MissAt_Half_mult + MissAt25k_mult + Bib,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_25k,newdata = testing) * predict(mod_lm_25,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 25km random forest multiplication model on the Final Training data
  final_training$lm_rato_25 <- final_training$Final_Time / predict(mod_lm_25,newdata = final_training)
  mod_rf_25_mult = train(lm_rato_25 ~ PaceAt5K + NrmPace_20K + NrmPace_Half + NrmPace_25K + MissAt20k_mult + MissAt_Half_mult + MissAt25k_mult + Bib,data=final_training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_25k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_25k_add.rds","mod_rf_25k_add.rds")
  mod_rf_25_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_25_mult.rds","mod_rf_25_mult.rds")
}

ggarrange(plot_rf_imp(mod_rf_25_mult,"Ratio Model"),plot_rf_imp(mod_rf_25k_add,"Residual Model"),ncol = 2, nrow = 1)
## It looks like we can drop the Bib and the pace at 5km for the 30km model


################################
##
## 30kM: Random Forest
## Per the feature importance at the 25km we will drop Bib and PaceAt5K
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At 25k the node size was 53, I'm going to drop 200 and add 75 and 150
  nodes <- c(10,25,50,75,100,150)
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_30k = train(lm_resids30 ~ NrmPace_20K + NrmPace_Half + NrmPace_25K + NrmPace_30K + MissAt20k_add + MissAt_Half_add + MissAt25k_add + MissAt30k_add,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_30k,newdata = testing) + predict(mod_lm_30,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_30k = train(lm_resids30 ~ NrmPace_20K + NrmPace_Half + NrmPace_25K + NrmPace_30K + MissAt20k_add + MissAt_Half_add + MissAt25k_add + MissAt30k_add,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_30k,newdata = testing) + predict(mod_lm_30,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}

  ## Train the final 30km additive random forest model on the Final Training data
  final_training$lm_resids30 <- final_training$Final_Time - predict(mod_lm_30,newdata = final_training)
  mod_rf_30k_add = train(lm_resids30 ~ NrmPace_20K + NrmPace_Half + NrmPace_25K + NrmPace_30K + MissAt20k_add + MissAt_Half_add + MissAt25k_add + MissAt30k_add,data=final_training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_30k = train(lm_rato_30 ~ NrmPace_20K + NrmPace_Half + NrmPace_25K + NrmPace_30K + MissAt20k_mult + MissAt_Half_mult + MissAt25k_mult + MissAt30k_mult,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_30k,newdata = testing) * predict(mod_lm_30,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_30k = train(lm_rato_30 ~ NrmPace_20K + NrmPace_Half + NrmPace_25K + NrmPace_30K + MissAt20k_mult + MissAt_Half_mult + MissAt25k_mult + MissAt30k_mult,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_30k,newdata = testing) * predict(mod_lm_30,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}
  
  ## Train the final 30km random forest multiplication model on the Final Training data
  final_training$lm_rato_30 <- final_training$Final_Time / predict(mod_lm_30,newdata = final_training)
  mod_rf_30_mult = train(lm_rato_30 ~ NrmPace_20K + NrmPace_Half + NrmPace_25K + NrmPace_30K +MissAt20k_mult + MissAt_Half_mult + MissAt25k_mult + MissAt30k_mult,data=final_training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_30k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_30k_add.rds","mod_rf_30k_add.rds")
  mod_rf_30_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_30_mult.rds","mod_rf_30_mult.rds")
}

ggarrange(plot_rf_imp(mod_rf_30_mult,"Ratio Model"),plot_rf_imp(mod_rf_30k_add,"Residual Model"),ncol = 2, nrow = 1)
## It looks like we can drop the MissAt20K, and the normalized pace at 20K and the Half from the 35km model


################################
##
## 35kM: Random Forest
## Per the feature importance at the 30km we will drop the MissAt20K and the Normalized Pace from 20K and the Half
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At 30k the node size was 113 so it looks like the node size needs to get larger again
  nodes <- c(25,50,75,100,150,175,200,500,1000)
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_35k = train(lm_resids35 ~ NrmPace_25K + NrmPace_30K + NrmPace_35K + MissAt_Half_add + MissAt25k_add + MissAt30k_add + MissAt35k_add,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_35k,newdata = testing) + predict(mod_lm_35,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_35k = train(lm_resids35 ~ NrmPace_25K + NrmPace_30K + NrmPace_35K + MissAt_Half_add + MissAt25k_add + MissAt30k_add + MissAt35k_add,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_35k,newdata = testing) + predict(mod_lm_35,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}
  
  ## Train the final 35km additive random forest model on the Final Training data
  final_training$lm_resids35 <- final_training$Final_Time - predict(mod_lm_35,newdata = final_training)
  mod_rf_35k_add = train(lm_resids35 ~ NrmPace_25K + NrmPace_30K + NrmPace_35K + MissAt_Half_add + MissAt25k_add + MissAt30k_add + MissAt35k_add,data=final_training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_35k = train(lm_rato_35 ~ NrmPace_25K + NrmPace_30K + NrmPace_35K + MissAt_Half_mult + MissAt25k_mult + MissAt30k_mult + MissAt35k_mult,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_35k,newdata = testing) * predict(mod_lm_35,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_35k = train(lm_rato_35 ~ NrmPace_25K + NrmPace_30K + NrmPace_35K + MissAt_Half_mult + MissAt25k_mult + MissAt30k_mult + MissAt35k_mult,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_35k,newdata = testing) * predict(mod_lm_35,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}
  
  ## Train the final 35km random forest multiplicative model on the Final Training data
  final_training$lm_rato_35 <- final_training$Final_Time / predict(mod_lm_35,newdata = final_training)
  mod_rf_35_mult = train(lm_rato_35 ~ NrmPace_25K + NrmPace_30K + NrmPace_35K + MissAt_Half_mult + MissAt25k_mult + MissAt30k_mult + MissAt35k_mult,data=final_training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_35k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_35k_add.rds","mod_rf_35k_add.rds")
  mod_rf_35_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_35_mult.rds","mod_rf_35_mult.rds")
}

ggarrange(plot_rf_imp(mod_rf_35_mult,"Ratio Model"),plot_rf_imp(mod_rf_35k_add,"Residual Model"),ncol = 2, nrow = 1)
## It looks like we can drop the MissAtHalf, MissAt25K and the normalized pace at 25K when we model at 40km


################################
##
## 40kM: Random Forest
## Per the feature importance at the 35km we will drop the MissAtHalf, MissAt25K and the Normalized Pace from 25K
## The "if (download_rfs){}" block is used to skip the time consuming model tuning and instead download a pre-optimized model
##
################################

if (download_rfs == "no") {
  ## Optimize a Random Forest by tuning the node-size 
  ## At 40km there is only ~2.2k left in the race so it isn't clear what ns is best so we'll keep the wide grid
  nodes <- c(25,50,75,100,150,175,200,500,1000)
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_40k = train(lm_resids40 ~ NrmPace_30K + NrmPace_35K + NrmPace_40K + MissAt30k_add + MissAt35k_add + MissAt40k_add,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_40k,newdata = testing) + predict(mod_lm_40,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_40k = train(lm_resids40 ~ NrmPace_30K + NrmPace_35K + NrmPace_40K + MissAt30k_add + MissAt35k_add + MissAt40k_add,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_40k,newdata = testing) + predict(mod_lm_40,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}
  
  ## Train the final 35km additive random forest model on the Final Training data
  final_training$lm_resids40 <- final_training$Final_Time - predict(mod_lm_40,newdata = final_training)
  mod_rf_40k_add = train(lm_resids40 ~ NrmPace_30K + NrmPace_35K + NrmPace_40K + MissAt30k_add + MissAt35k_add + MissAt40k_add,data=final_training, method = "rf",nodesize = ns,ntree = nt)
  
  
  ## Now repeat the same process for the multiplicative error term calculation
  setup_for_rf_tune_for()
  for (ns in nodes){
    rf_40k = train(lm_rato_40 ~ NrmPace_30K + NrmPace_35K + NrmPace_40K + MissAt30k_mult + MissAt35k_mult + MissAt40k_mult,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_40k,newdata = testing) * predict(mod_lm_40,testing)
    rmses = append(rmses,RMSE(testing$Final_Time,preds))
  }
  
  setup_for_rf_tune_while(nodes,rmses)
  while (new_rmse < (old_rmse*0.99)) {
    if (new_rmse >0) {old_rmse = new_rmse}    ## For the first loop we don't have a refresh of the old rmse
    nt = old_nt + 25
    rf_40k = train(lm_rato_40 ~ NrmPace_30K + NrmPace_35K + NrmPace_40K + MissAt30k_mult + MissAt35k_mult + MissAt40k_mult,data=training, method = "rf",nodesize = ns,ntree = nt)
    preds = predict(rf_40k,newdata = testing) * predict(mod_lm_40,testing)
    new_rmse = RMSE(testing$Final_Time,preds)
  }
  ## If RMSE increased in the last step of the loop we will go back one
  if (new_rmse > old_rmse) { nt = nt - 25}
  
  ## Train the final 40km random forest multiplication model on the Final Training data
  final_training$lm_rato_40 <- final_training$Final_Time / predict(mod_lm_40,newdata = final_training)
  mod_rf_40_mult = train(lm_rato_40 ~ NrmPace_30K + NrmPace_35K + NrmPace_40K + MissAt30k_mult + MissAt35k_mult + MissAt40k_mult,data=final_training, method = "rf",nodesize = ns,ntree = nt)
} else {
  mod_rf_40k_add <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_40k_add.rds","mod_rf_40k_add.rds")
  mod_rf_40_mult <- read_rds_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/mod_rf_40_mult.rds","mod_rf_40_mult.rds")
}

ggarrange(plot_rf_imp(mod_rf_40_mult,"Ratio Model"),plot_rf_imp(mod_rf_40k_add,"Residual Model"),ncol = 2, nrow = 1)
# There isn't another iteration so this plot isn't really important


################################
##
## Validation
## Download the validation data and see how the model performs!
##
################################

## Download the data, clean and pre-process it
validation <- read_csv_from_github("https://raw.githubusercontent.com/delucj/BostonMarathonTimeProjections/main/marathon_results_2017.csv","marathon_results_2017.csv")

validation <- pre_process_data(validation,2017)
validation <- add_mid_race_miss(validation)

validation_naive <- get_naive_performance(validation)
validation_lm <- get_linear_performance(validation)
validation_smart <- get_rf_performance(validation)
validation_smart$Improvement <- 100 * (validation_naive$Naive.RMSE - validation_smart$RF.RMSE) / validation_naive$Naive.RMSE

# Plot the naive mode RMSE at each split
overall <- ggplot() + geom_point(aes(x=Distance,y=Naive.RMSE),data = validation_naive,color="black") +
	geom_line(aes(x=Distance,y=Naive.RMSE),data = validation_naive,color="black") +
	geom_point(aes(x=Distance,y=RF.RMSE),data = validation_smart,color="blue") +
	geom_line(aes(x=Distance,y=RF.RMSE),data = validation_smart,color="blue") +
  geom_point(aes(x=Distance,y=Linear.RMSE),data = validation_lm,color="red") +
	geom_line(aes(x=Distance,y=Linear.RMSE),data = validation_lm,color="red") +
	xlab("Race Distance Completed (km)") + ylab("Model RMSE (minutes)") + 
	ggtitle("Model Validation | 2017") +
	annotate("text",x=27,y=34,label = "BAA Baseline Model",color="black") +
	annotate("text",x=27,y=32,label = "Linear Model",color="red") +
  annotate("text",x=27,y=30,label = "Lm and RF Ensemble Model",color="blue") +
  xlim(c(0,42)) + ylim(c(0,35)) + 
  geom_rect(aes(xmin = 18,xmax = 27, ymin = 8, ymax = 24),
		color = "black",fill = "light green", alpha=0.2)

improved <- ggplot() + geom_point(aes(x=Distance,y=Improvement),data = validation_smart,color="black") +
	geom_line(aes(x=Distance,y=Improvement),data = validation_smart,color="black") +
	xlab("Race Distance Completed (km)") + ylab("Improvement over Baseline (%)") + 
  xlim(c(0,42)) + ylim(c(0,100)) + 
  geom_rect(aes(xmin = 18,xmax = 27, ymin = 40, ymax = 55),
		color = "black",fill = "light green", alpha=0.2) +
  annotate("text",x=22.5,y=60,
		label = "Key Prediction Check-Points",color="black") +
  ggtitle("Improvement over Baseline")

## Display the plot
ggarrange(overall,improved,ncol = 2, nrow = 1)

## The start_time and end_time show how long it took for the script to execute
end_time = Sys.time()
start_time
end_time