# ISLR-College-Data-Exploration-in-R
## Introduction
In this project, we are going to showcase R using the College dataset provided in the ISLR package! Using the ggplot2 package to visualize this data, we will explore enrollments and graduation rates between public and private institutes. Let's go ahead and load in our data and take a look at it:
```
library("ggplot2")
library(ISLR)
df <- data.frame(College)
head(df)
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College1.png?raw=True)
```
structure(df)
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College2.png?raw=True)
```
summary(df)
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College3.png?raw=True)
## Data Visualization
We can start off with a simple scatterplot of Grad.Rate versus Room.Board, and we will color it by the Private column:
```
ScatterDF <- ggplot(data=df, aes(x=Grad.Rate, y=Room.Board, color=Private)) + geom_point()
ScatterDF
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College4.png?raw=True)
The scatter plot has very little correlation between both private and non private groups when it comes to grad rate versus room and board. Both groups appear to merge and go in a particular direction. Perhaps if you are being picky, you could say the data tends to clump at around the 55-60 grad rate and 4000 room board range. Now, let's create a histogram of full-time undergraduate students colored by private:
```
HistogramDF <- ggplot(data=df, aes(x=F.Undergrad, fill=Private)) + geom_histogram()
HistogramDF
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College5.png?raw=True)
The data here is heavily skewed to the right. We see a peak at about 2000 full time undergraduates and outliers for non private full time undergraduates for the higher volume of students. Lets try another histogram of Grad.Rate colored by private:
```
HistogramDF2 <- ggplot(data=df, aes(x=Grad.Rate, fill=Private)) + geom_histogram()
HistogramDF2
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College6.png?raw=True)
The histogram here shows a normal distribution with a mean of about 60-65. The strange thing about this graph is that we have a grad rate that is above 100 percent. If we look at the data frame and filter the Grad Rate column from high to low, we see that a value of 118 has been placed for Cazenovia College. Let's go ahead and set that to 100
```
>df$Grad.Rate <- replace(df$Grad.Rate, df$Grad.Rate>100, 100)
```
## Hypothesis Testing
Let's test the difference of the mean enrollment between the private and non-private colleges. Lets define Ho as: There is no difference between the mean enrollment between the private and non private colleges. Let's define Ha as: There is a difference between the mean enrollment between the private and non private colleges. 
```
res <- t.test(Enroll~Private, data=df)
res
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College7.png?raw=True)
Using a t-test to test the difference of means between enrollment in private and enrollment in non private, we get a p value significantly less than 0.05. This means that we reject the null hypothesis, and can say that there is a difference between the mean enrollment between private and non private colleges. Now, let's test the difference of the mean graduation rates between the private and non-private colleges. We will define Ho as: There is no difference between the mean graduation rate between the private and non private colleges. We will define Ha as: There is a difference between the mean graduation rate between the private and non private colleges.
```
res2 <- t.test(Grad.Rate~Private, data=df)
res2
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College8.png?raw=True)
Using a t-test to test the difference of means between graduation rate in private and enrollment in non private, we get a p value significantly less than 0.05. This means that we reject the null hypothesis, and can say that there is a difference between the mean graduation rate between private and non private colleges.

## Regression
Let's split our data into training and testing sets 70/30. We will set the seed as 101:
```
library(caTools)
set.seed(101)
split = sample.split(df$Private, SplitRatio = .70)
final.train = subset(df,split==TRUE)
final.test = subset(df,split==FALSE)
```
Using the caTools library, let's build a logistic regression model to predict whether or not a school is Private:
```
final.log.model <- glm(formula=Private ~ . , family = binomial(logit), data = final.train)
summary(final.log.model)
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College9.png?raw=True)
Now let's use predict() to predict the Private label on the test data:
```
glm_probs = data.frame(probs = predict(final.log.model, type="response"))
head(glm_probs)
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College10.png?raw=True)
Great! The head here is showing some pretty high probabilities. Now let's use the caTools library to build a multiple linear regression model to predict the enrollment:
```
split = sample.split(df$Enroll, SplitRatio = .70)
final.train = subset(df, split == TRUE)
final.test = subset(df, split == FALSE)
regressor <- lm(Enroll ~ ., data = final.train)
summary(regressor)
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College11.png?raw=True)
As shown above in the summary, the model shoes an R-Squared value of .95. Let's go ahead and predict the enrollments on test data:
```
y_pred <- predict(regressor, newdata = final.train)
head(y_pred)

original <- final.train$Enroll
predicted <- y_pred
d <- original - predicted
mse = mean((d)^2)
R2 = 1 - (sum((d)^2) / sum((original - mean(original))^2))

cat("MSE:", mse, "\n", "R-squared:", R2, "\n")
```
![alt text](https://github.com/SpencerSewell/Pictures/blob/main/College12.png?raw=True)
We have a very high r squared value! This model worked well for our prediction!
