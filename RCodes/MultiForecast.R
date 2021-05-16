library(xgboost)
library(tidymodels)
library(modeltime)
library(tidyverse)
library(lubridate)
library(timetk)
library(ggthemes)


## Read the aggregated Dataset From Original Data FIle ####
data = read.csv("/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/WeeklyCorona.csv")

  
  karachidata <- data
  karachidata$test_date = as_date(karachidata$test_date)
  print(karachidata)
  
  
  ## Split Data Set in 80 20 ratio For train and test ##
  splits <- initial_time_split(karachidata, prop = 0.8)
  
  
  ### Fitting Training Data To all Models ####
 ### With equation corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache ####
  
  
  model_fit_arima_no_boost <- arima_reg() %>%
    set_engine(engine = "auto_arima") %>%
    fit(corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache, data = training(splits))
  
  
  # Model 2: arima_boost ----
  model_fit_arima_boosted <- arima_boost(
    min_n = 2,
    learn_rate = 0.015
  ) %>%
    set_engine(engine = "auto_arima_xgboost") %>%
    fit(corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache + as.numeric(test_date) + factor(month(test_date, label = TRUE), ordered = F),
        data = training(splits))
  
  
  
  model_fit_ets <- exp_smoothing() %>%
    set_engine(engine = "ets") %>%
    fit(corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache, data = training(splits))
  
  
  
  
  model_fit_prophet <- prophet_reg() %>%
    set_engine(engine = "prophet") %>%
    fit(corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache, data = training(splits))
  
  
  
  model_fit_lm <- linear_reg() %>%
    set_engine("lm") %>%
    fit(corona_result ~ cough+fever+sore_throat+shortness_of_breath+head_ache+as.numeric(test_date) + factor(month(test_date, label = TRUE), ordered = FALSE),
        data = training(splits))
  
  
  
  model_spec_mars <- mars(mode = "regression") %>%
    set_engine("earth") 
  
  recipe_spec <- recipe(corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache, data = training(splits)) %>%
    step_date(test_date, features = "month", ordinal = FALSE) %>%
    step_mutate(date_num = as.numeric(test_date)) %>%
    step_normalize(date_num) %>%
    step_rm(test_date)
  wflw_fit_mars <- workflow() %>%
    add_recipe(recipe_spec) %>%
    add_model(model_spec_mars) %>%
    fit(training(splits))
  
  
  
  ## Machine Learning
  
  recipe_spec <- recipe(corona_result ~ test_date+cough+fever+sore_throat+shortness_of_breath+head_ache, training(splits)) %>%
    step_timeseries_signature(test_date) %>%
    step_rm(contains("am.pm"), contains("hour"), contains("minute"),
            contains("second"), contains("xts")) %>%
    step_fourier(test_date, period = 365, K = 5) %>%
    step_dummy(all_nominal())
  
  recipe_spec %>% prep() %>% juice()
  
  
  model_spec_glmnet <- linear_reg(penalty = 0.01, mixture = 0.5) %>%
    set_engine("glmnet")
  
  workflow_fit_glmnet <- workflow() %>%
    add_model(model_spec_glmnet) %>%
    add_recipe(recipe_spec %>% step_rm(test_date)) %>%
    fit(training(splits))
  
  model_spec_rf <- rand_forest(trees = 500, min_n = 50) %>%
    set_engine("randomForest")
  
  workflow_fit_rf <- workflow() %>%
    add_model(model_spec_rf) %>%
    add_recipe(recipe_spec %>% step_rm(test_date)) %>%
    fit(training(splits))
  
  
  model_spec_prophet_boost <- prophet_boost(seasonality_yearly = TRUE) %>%
    set_engine("prophet_xgboost") 
  
  workflow_fit_prophet_boost <- workflow() %>%
    add_model(model_spec_prophet_boost) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))
  
  workflow_fit_prophet_boost
  
  ### Representing All trained Models In One Table ####
  
  model_table <- modeltime_table(
    model_fit_arima_no_boost,
    model_fit_arima_boosted,
    model_fit_prophet,
    model_fit_ets,
    model_fit_lm,
    wflw_fit_mars,
    workflow_fit_glmnet,
    workflow_fit_rf,
    workflow_fit_prophet_boost
  )
  
  
  ### Claibrating Model Table Over Test Data ###
  model_table
  calibration_table <- model_table %>%
    modeltime_calibrate(testing(splits))
  
  
  
  ### Forecasting All Models In Table on 20% Test Data  #####
  
  calibration_table %>%
    modeltime_forecast(actual_data = karachidata,new_data = testing(splits))
  
  ### Get Error Values Of All Models With Different Error Metrics(ie: MAPE,MASE,MAE) ###
  
  forecast <- calibration_table %>%
    modeltime_accuracy() 
  
  ### Saving Error Table With Sorted MAPE values. Model with least MAPE at top ####
  
  ordered <- forecast[order(forecast$mape),]
  write.csv(ordered,"/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/WeeklyCoronaAcc.csv")
  
  
  ## Get Forecast Image By Best Performing Algorithm on Test DAta
  calibration_table1 <- model_table[ordered[1:1,]$.model_id,] %>%
    modeltime_calibrate(testing(splits))
  image<- calibration_table1 %>%
    modeltime_forecast(actual_data = karachidata,new_data = testing(splits)) %>%
    plot_modeltime_forecast(.interactive = FALSE)
  png(width = 780, height = 680, units = "px",filename = "/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/WeeklyCorona1.png")
  plot(image)
  dev.off()
  
  
  ## Get Largest Date From Dataset
  max_date =max(as_date(karachidata$test_date))
  max_date
  
  
  ### Read Multi Variate Data For Future Forecasting ###
  test_date <- seq(as.Date(max_date)+5, by = 'week', length.out = 13)
  df = read.csv("/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/WeeklyCoronaTest.csv")
  df = data.frame(test_date,df)
  print("########################")
  print(df)
  print("########################")
  
  
  ## Forecast On Future Multi Var Symptom Count Data With Best Performing Model ###
  ## Get Forecast Image ###
  image <- calibration_table %>%
    filter(.model_id == ordered[1:1,]$.model_id) %>%
    modeltime_refit(karachidata) %>%
    modeltime_forecast(actual_data = karachidata,new_data = df) %>%
    plot_modeltime_forecast(.interactive = FALSE)
  png(width = 780, height = 680, units = "px",filename = "/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/WeeklyCoronaForecast1.png")
  plot(image)
  dev.off()
  
  ## Get Forecast Values For Future ###
  Fdata <- calibration_table %>%
    filter(.model_id == ordered[1:1,]$.model_id) %>%
    modeltime_refit(karachidata) %>%
    modeltime_forecast(actual_data = karachidata,new_data = df)
  write.csv(Fdata,"/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/WeeklyCoronaTestForecast.csv")



