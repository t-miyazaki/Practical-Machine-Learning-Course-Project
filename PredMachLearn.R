# set up
library(tidyverse)
library(caret)

# loading data
training <- read.csv('pml-training.csv')
testing <- read.csv('pml-testing.csv')
str(training)
summary(training)

# data pre-processing
x_train <- select(training, 
                  ends_with(c("_x","_y","_z")),
                  starts_with(c("roll_","pitch_","yaw_")))
y_train <- as.factor(training$classe)
x_test <- select(testing,
                 ends_with(c("_x","_y","_z")),
                 starts_with(c("roll_","pitch_","yaw_")))

# random forests with a parallel implementation
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(0)
fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
modFit <- train(x_train, y_train, method="rf", data = training, trControl = fitControl)

stopCluster(cluster)
registerDoSEQ()

modFit
modFit$finalModel
predict(modFit, x_test)
