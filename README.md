## Logistic Regression Model both in Python and R

# I used logistic Regression to classify the Social_Media_Ads dataset , and produce two Graphs of training set and test set each so We can do the Analysis properly.

Binary Logistic Regression is a special type of regression where binary response variable is related to a set of explanatory variables, which can be discrete and/or continuous. The important point here to note is that in linear regression, the expected values of the response variable are modeled based on combination of values taken by the predictors. In logistic regression Probability or Odds of the response taking a particular value is modeled based on combination of values taken by the predictors.

### Logistic regression is applicable, if:

- we want to model the probabilities of a response variable as a function of some explanatory variables, e.g. "success" of admission as a function of gender.
- we want to perform descriptive discriminate analyses such as describing the differences between individuals in separate groups as a function of explanatory variables, e.g. student admitted and rejected as a function of gender
- we want to predict probabilities that individuals fall into two categories of the binary response as a function of some explanatory variables, e.g. what is the probability that a student is admitted given she is a female
- we want to classify individuals into two categories based on explanatory variables, e.g. classify new students into "admitted" or "rejected" group depending on their gender.

# Remember in Scikit learn Model , from 0.18 version test_train_split Class is not imported from Cross_Validation instead through model_selection in Python
