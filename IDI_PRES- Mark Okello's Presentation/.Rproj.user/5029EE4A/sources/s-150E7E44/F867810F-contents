# Libraries We shall be Using
library(caTools) # Splitting the dataset into train and test set
library(ggplot2) # Visualizing Results
library(scales) # continous scale on axis
library(readxl) # Reading execl data
library(class) # Fitting KNN Algorithm
library(e1071) # Fitting Support vector machine, Naive Bayes
library(rpart) # Fitting Decision Tree Classifier
library(randomForest) # Fitting Random Forest 
library(cluster) # Visualizing kmeans clusters and HC 

# ................REGRESSION SECTION.....................

# LINEAR REGRESSION
# simple linear regression 
# Loading our dataset 
dataset <- read.csv('data/crafted_umeme-dataset.csv')
# Seeing what our data is like
head(dataset,5)
# Subsetting only the Variables(columns) for our model
slr_data <- dataset[ ,1:2]
# View(slr_data) # if interested to see what it looks like

# Splitting the dataset into Training set and Test set
set.seed(772018)
dividing <- sample.split(slr_data$SalaryAyear, SplitRatio = 0.8)
training_set <- subset(slr_data, dividing == TRUE)
test_set <- subset(slr_data, dividing == FALSE)

# Fitting the model
regressionModel = lm(formula = SalaryAyear ~ YearsExperience,
                     data = training_set)

# Predicting Results
predicting = predict(regressionModel, newdata = test_set)

# Visualising the Training set results
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$SalaryAyear),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressionModel, newdata = training_set)),
            colour = 'blue') +
  scale_y_continuous(labels = comma)+
  ggtitle(paste("Salary Vs Years of Experience")) +
  labs(x = "Years of Experience",y = "Salary",
       caption = "Source of Data: Crafted")

# Visualising the Test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressionModel, newdata = training_set)),
            colour = 'blue') +
  scale_y_continuous(labels = comma)+
  ggtitle(paste("Salary Vs Years of Experience")) +
  labs(x = "Years of Experience",y = "Salary",
       caption = "Source of Data: Crafted")

# Multiple linear regression 

# Loading the neccessary data
mlr_data <- dataset[ ,3:6]
head(mlr_data)

# Factoring/encoding categorical data using integer encoding
mlr_data$Branch <- factor(mlr_data$Branch,
                         levels = c('Kampala', 'Mbarara', 'Soroti'),
                         labels = c(1, 2, 3))

set.seed(772018)
dividing_mlr <- sample.split(mlr_data$ProfitMade, SplitRatio = 0.8)
training_set <- subset(mlr_data, dividing_mlr == TRUE)
test_set <- subset(mlr_data, dividing_mlr == FALSE)

# Fitting model to Training set
regressionModel_mlr = lm(formula = ProfitMade ~ .,
                         data = training_set)

# Predicting the Test set results
predicting = predict(regressionModel_mlr, newdata = test_set)

# Backward Elimination for an optimal model
b_elimination = lm(formula = ProfitMade ~ DamagesPaid + AdministrationCost + Branch,
                   data = mlr_data)
summary(b_elimination)

b_elimination = lm(formula = ProfitMade ~ DamagesPaid + AdministrationCost + factor(Branch, exclude = 3),
                   data = mlr_data)
summary(b_elimination)

b_elimination = lm(formula = ProfitMade ~ DamagesPaid  + factor(Branch, exclude = 3),
                   data = mlr_data)
summary(b_elimination)

# Polynomial Regression
pr <- slr_data

# Fitting Polynomial Regression Model to the dataset
pr$YearsExperience2 = pr$YearsExperience^2
pr$YearsExperience3 = pr$YearsExperience^3
pr$YearsExperience4 = pr$YearsExperience^4
regressionModel_pr = lm(formula = SalaryAyear ~ .,
                        data = pr)

# Visualising the Polynomial Regression results with new variables
ggplot() +
  geom_point(aes(x = pr$YearsExperience, y = pr$SalaryAyear),
             colour = 'red') +
  geom_line(aes(x = pr$YearsExperience, y = predict(regressionModel_pr, newdata = pr)),
            colour = 'blue') +
  scale_y_continuous(labels = comma)+
  ggtitle(paste("Polynomial Regression")) +
  labs(x = "Years of Experience",y = "Salary",
       caption = "Source of Data: Crafted")

# smoother curve with new variables from powers

x_grid = seq(min(pr$YearsExperience), max(pr$YearsExperience), 0.2)
ggplot() +
  geom_point(aes(x = pr$YearsExperience, y = pr$SalaryAyear),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressionModel_pr,
                                        newdata = data.frame(YearsExperience = x_grid,
                                                             YearsExperience2 = x_grid^2,
                                                             YearsExperience3 = x_grid^3,
                                                             YearsExperience4 = x_grid^4))),
            colour = 'blue') +
  scale_y_continuous(labels = comma)+
  ggtitle('Polynomial Regression') +
  labs(x = "Years of Experience",y = "Salary",
       caption = "Source of Data: Crafted")


# ........................Classification..........................

# Logistic Regression

# Loading the data
airT_data <- read_excel('data/airtime_borrowing.xls')

# Encoding paid as factor
airT_data$Paid <- factor(airT_data$Paid, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
set.seed(772018)
dividing <- sample.split(airT_data$Paid, SplitRatio = 0.8)
training_set <- subset(airT_data, dividing == TRUE)
test_set <- subset(airT_data, dividing == FALSE)

# Feature Scaling
training_set[-3] <- scale(training_set[-3])
test_set[-3] <- scale(test_set[-3])

# Fitting the model
lr_classifier = glm(formula = Paid ~ .,
                    family = binomial,
                    data = training_set)

# Predicting results
t_prediction = predict(lr_classifier, 
                       type = 'response', newdata = test_set[-3])
round_prediction = ifelse(t_prediction > 0.5, 1, 0)

# Confusion Matrix
cm = table(test_set$Paid, round_prediction > 0.5)
cm

# K-Nearest Neighbors (K-NN)

# Importing the dataset
# We are using the same dataset. Only thing is to fit the model

# Fitting K-NN and Predicting results
knn_classifier = knn(train = training_set[, -3],
                     test = test_set[, -3],
                     cl = training_set$Paid,
                     k = 5,
                     prob = TRUE)

# Confusion Matrix
cm = table(test_set$Paid, knn_classifier)
cm

# Support Vector Machine Classification

# Not Loading the dataset, Encoding or factoring, Splitting the dataset into the Training set and Test set and  Feature Scaling

# Fitting Kernel SVM model

svm_classifier = svm(formula = Paid ~ .,
                     data = training_set,
                     type = 'C-classification',
                     kernel = 'radial') # can be done without the kernel

# Predicting results
predicting = predict(svm_classifier, newdata = test_set[-3])

# Confusion Matrix
cm = table(test_set$Paid, predicting)
cm

# Naive Bayes

# Fitting Naive Bayes model

naive_classifier = naiveBayes(x = training_set[-3],
                              y = training_set$Paid)

# Predicting results
predicting = predict(naive_classifier, newdata = test_set[-3])

# Confusion Matrix
cm = table(test_set$Paid, predicting)
cm

# Decision Tree Classification

# Fitting Decision Tree Classification

dtc_classifier = rpart(formula = Paid ~ .,
                       data = training_set)

# Predicting results
predicting = predict(dtc_classifier, newdata = test_set[-3], type = 'class')

# Confusion Matrix
cm = table(test_set$Paid, predicting)
cm

# Random Forest Classification

# Fitting Random Forest Classification
set.seed(2018)
rf_classifier = randomForest(x = training_set[-3],
                             y = training_set$Paid,
                             ntree = 10)

# Predicting results
predicting = predict(rf_classifier, newdata = test_set[-3])

# Confusion Matrix
cm = table(test_set$Paid, predicting)
cm

# Adding changing number of trees
set.seed(2018)
rf_classifier = randomForest(x = training_set[-3],
                             y = training_set$Paid,
                             ntree = 300)

# Predicting results
predicting = predict(rf_classifier, newdata = test_set[-3])

# Confusion Matrix
cm = table(test_set$Paid, predicting)
cm

# Change number of trees until optimal

# .....................Clustering

# K-Means Clustering

# Importing the dataset
creditcard_data <- read.csv('data/CC GENERAL.csv')
head(creditcard_data, 5)
creditcard_data <- creditcard_data[c(2,3)] # 2:3

# Using the elbow method to find good number of clusters
set.seed(772018)
weighted_sum = vector()
for (series in 1:15) weighted_sum[series] = sum(kmeans(creditcard_data, series)$withinss)
plot(1:15,
     weighted_sum,
     type = 'b',
     main = paste('Clusters using Elbow Method'),
     xlabs = 'Number of clusters', 
     ylabs = 'weighted_sum')

# Fitting K-Means model
set.seed(18)
kmeans_cluster = kmeans(x = creditcard_data, centers = 2)
clustering_kmeans = kmeans_cluster$cluster

# Visualising the clusters
clusplot(creditcard_data,
         clustering_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 1,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of Clients Behavior'),
         xlab = 'Account Balance',
         ylab = '% Balance Enquiries')

# Hierarchical Clustering

# Importing the dataset # Using the same data for Kmeans clustering

# Using the dendrogram to find get optimal clusters
dendrogram = hclust(d = dist(creditcard_data, method = 'euclidean'), 
                    method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = ' Length')

# Fitting Hierarchical Clustering model
hc_model = hclust(d = dist(creditcard_data, method = 'euclidean'),
                  method = 'ward.D')
clustering_hc = cutree(hc_model, 2)

# Visualising the clusters
clusplot(creditcard_data,
         clustering_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 1,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of client Behavior'),
         xlab = 'Account Balance',
         ylab = '% Balance Enquiries')
