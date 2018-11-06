---
title: "Practical Machine Learning"
author: "Edwin Wanner"
date: "30 oktober 2018"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(caret)
```

## Overview of project
Devices like 'Jawbone Up', 'Fuelband' and 'Fitbit' collect large amount of data about personal activity. These type of devices are part of the quantified self movement - a group that takes measurements about themselves regularly to improve their health and to find patterns. The goal of this project is to use the data of the devices on the belt, forearm, arm and dumbell of 6 participants to predict the manner in which they did the exercises. In this analysis, three algorithms have been chosen to investigate: Classification trees, Random Forests and Gradient Boosting Models. All of this will be done using cross-validation.

## Loading the data
The data is loaded from the following webpages:

- Training data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

- Testing data: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

```{r loading data}
trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingURL, destfile = 'training.csv', mode = 'wb')
download.file(testingURL, destfile = 'testing.csv', mode = 'wb')

training <- read.csv(file = "training.csv", header=TRUE, sep = ",")
testing <- read.csv(file = "testing.csv", header = TRUE, sep = "," )
```

## Data preprocessing
Looking at the data, the data has a lot of NA and blank values. We will remove those variables which have 95% blank or NA values. These variables add too little value.

```{r blank and na}
colSums(is.na(training) | training == "")
blank_ids <- which(apply(training, 2,function(x) sum(x == "")) > 0.95*length(training[,1]))
NA_ids <- which(apply(training, 2,function(x) sum(is.na(x))) > 0.95*length(training[,1]))
trainingCleaned <- training[,-c(blank_ids,NA_ids)]
dim(trainingCleaned)

blank_ids_test <- which(apply(testing, 2,function(x) sum(x == "")) > 0.95*length(testing[,1]))
NA_ids_test <- which(apply(testing, 2,function(x) sum(is.na(x))) > 0.95*length(testing[,1]))
testingCleaned <- testing[,-c(blank_ids_test, NA_ids_test)]
dim(testingCleaned)
```

Now that we have removed the variables which are blank or NA, we have 60 variables left.
Looking at the remaining variables, we see that the first 7 columns are not relevant ('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', ' cvtd_timestamp', 'new_window','num_window'). These columns are removed.

```{r remove irrelevant}
trainingCleaned <- trainingCleaned[,-c(1:7)]
testingCleaned <- testingCleaned[,-c(1:7)]
``` 
Next, we will create a separate training and test set from the current dataset.

```{r training test split}
set.seed(123)
inTraining <- createDataPartition(trainingCleaned$classe, p = 0.8, list = FALSE)
train_set <- trainingCleaned[inTraining,]
test_set <- trainingCleaned[-inTraining,]
dim(train_set)
dim(test_set)
```
## Model building
Now that we have got our training- and test set, a number of models will be trained:

1. Classification tree
2. Random forest (multiple classification trees)
3. Gradient Boosting Models

To prevent overfitting, 5-fold cross-validation is applied. The number 5 is chosen as it is frequently chosen when performing cross-validation.



### Classification tree
```{r classification tree, cache = TRUE}
control <- trainControl(method = "cv", number = 5, allowParallel = T)
fit_CT <- train(classe~., data = train_set, method = "rpart", trControl = control)
predict_CT <- predict(fit_CT, newdata = test_set)
cm <- confusionMatrix(test_set$classe, predict_CT)
cm$overall[1]
```
The classification tree, trained with 5-fold cross-validation, resulted in an accuracy of 49.7%.

### Random Forest
Next, we will create a random forest. Again, 5-fold cross-validation is applied.

```{r RF, cache = TRUE}
fit_RF <- train(classe~., data = train_set, method = "rf", verbose = FALSE,trControl = control)
predict_RF <- predict(fit_RF, newdata = test_set)
confusionMatrix(test_set$classe, predict_RF)
```
Some graphs of the Random Forest show that the error decreases rapidly and stabilizes after creating around 60 trees.

```{r rf graphs, cache = TRUE}
plot(fit_RF$finalModel)
```

The Random forest, trained with 5-fold cross-validation, resulted in an accuracy of 99.4% for the test set. This is a major increase in the accuracy compared to the classification tree.

### Gradient Boosting
```{r GBM, cache = TRUE}
fit_GBM <- train(classe~., data = train_set, method = "gbm", verbose = FALSE,trControl = control)
predict_GBM <- predict(fit_GBM, newdata = test_set)
cm_GBM <- confusionMatrix(test_set$classe, predict_GBM)
cm_GBM
```
Some plots of the model show that using a higher tree depth provides a higher accuracy. Furthermore, using more boosting iterations could get you an even higher accuracy.

```{r gbm plot}
plot(fit_GBM)
```

The Gradient Boosting model, trained with 5-fold cross-validation, resulted in an accuracy of 96.5%.

## Conclusion
Looking at the results of the three models, one can conclude that the random forest model performed best with an accuracy of 99.4%.
The best model is used to predict the outcome of the testing dataset.

```{r testing}
predict_testing <- predict(fit_RF, newdata = testingCleaned)
predict_testing
```