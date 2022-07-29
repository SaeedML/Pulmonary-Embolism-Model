library(ROSE)
library(dplyr)
library(h2o)

df = read.csv("/Users/saeed/Desktop/Data.csv")
############### Min-Max scaler #############################################
min_max <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

df$Demog.Age = min_max(df$Demog.Age)
df$ClinicFind.SBP = min_max(df$ClinicFind.SBP)
df$ClinicFind.HR = min_max(df$ClinicFind.HR)
df$ClinicFind.RR = min_max(df$ClinicFind.RR)
df$Symptoms.O2.sat = min_max(df$Symptoms.O2.sat)
df$X1stLab.Hb = min_max(df$X1stLab.Hb)
df$X1stLab.Cr = min_max(df$X1stLab.Cr)
df$X1stLab.Plt = min_max(df$X1stLab.Plt)
df$X1stLab.WBC = min_max(df$X1stLab.WBC)
df$Lab.hsTnT = min_max(df$Lab.hsTnT)
df$ECG.HR = min_max(df$ECG.HR)
df$Echo.LVEF = min_max(df$Echo.LVEF)

##################### Train-Test Splitting ################################
sam = sample(x = nrow(df), size = (nrow(df)*0.60))
train = df[sam,]
test = df[-sam,]

train = ROSE(Composite.Outcome~. , data = train, N = 1000)$data
###################### H2O initiation and conversion of datasets ##########
h2o.init()
x = c(1:85)
y = 86
train = as.h2o(train)
test = as.h2o(test)
####################### Train-Valid Splitting #############################
splits <- h2o.splitFrame(data = train, 
                         ratios = c(0.75),
                         seed = 1)
train <- splits[[1]]
valid <- splits[[2]]

###################### Gradient Boosting Model ############################

gbm_params <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))

gbm_grid <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params)
gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                             sort_by = "auc",
                             decreasing = TRUE)
best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])

best_gbm_perf <- h2o.performance(model = best_gbm,
                                  newdata = test)
best_gbm_perf

###################### Neural Network Model ################################
activation_opt <- c("Rectifier", "Maxout", "Tanh")

l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = valid,
                    seed = 1,
                    hidden = c(10,10),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "AUC", 
                           decreasing = TRUE)
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)
best_dl_perf <- h2o.performance(model = best_dl, newdata = test)
best_dl_perf

plot(best_dl, 
     timestep = "epochs", 
     metric = "auc")
     
######################### Logistic Regression Model ##########################

glm_params <- list(alpha= c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

glm_grid <- h2o.grid("glm", x = x, y = y,
                      grid_id = "glm_grid",
                      training_frame = train,
                      validation_frame = valid,
                      seed = 1,
                      hyper_params = glm_params)
glm_gridperf <- h2o.getGrid(grid_id = "glm_grid",
                             sort_by = "logloss",
                             decreasing = FALSE)
best_glm <- h2o.getModel(glm_gridperf@model_ids[[1]])
best_glm_perf <- h2o.performance(model = best_glm, newdata = test)
best_glm_perf
