# Megaline Prediction Model - Subscribers Behaviour Analysis

**Data Analyst Bootcamp** @ Tripleten<br>
Sprint 11 Project - Intro to Machine Learning

## Context

At Megaline, a leading mobile carrier, we are committed to enhancing customer satisfaction by ensuring our subscribers are on the most suitable plan for their needs. After the introduction of our newer plans, Smart and Ultra, we have discovered that many of our customers continue to use legacy plans. To address this, we have initiated a program to transition those customers to the newer, optimized plans. To support this effort, the Customer Insights Department aims to develop a predictive model to recommend either the Smart or Ultra plan based on user behavior.

## Overview
We developed a classification model to predict the most appropriate plan for each user, based on monthly behavior data of subscribers who have already switched to the new plans. The dataset includes metrics such as the number of calls made, call duration, number of text messages sent, the internet traffic used, and the plan for the current month.

This project involved splitting the data into training, validation and test sets, experimenting with various models and hyperparameters, and evaluating the models' performance. As a requirement, we have set the threshold for accuracy at a minimum of 75%. Lastly, we conducted a sanity check to ensure the model's robustness. The ultimate objective of this project is to deliver a high-accuracy model that will facilitate personalized plan recommendations, thus improving customer satisfaction and loyalty.

## Libraries/Packages Used

The following libraries and packages were used in this project:

- **Pandas**: Data manipulation and analysis (`pandas`)
- **Seaborn**: Data visualization (`seaborn`)
- **Matplotlib**: Plotting library (`matplotlib`)
- **NumPy**: Numerical computing (`numpy`)

### Scikit-learn (sklearn)

- **Train Test Split**: For splitting the dataset into training and testing sets (`sklearn.model_selection.train_test_split`)
- **Decision Tree Classifier**: Decision tree classification algorithm (`sklearn.tree.DecisionTreeClassifier`)
- **Random Forest Classifier**: Ensemble method for classification (`sklearn.ensemble.RandomForestClassifier`)
- **Logistic Regression**: Logistic regression algorithm (`sklearn.linear_model.LogisticRegression`)
- **Metrics**: Evaluation metrics such as accuracy, precision, recall, and F1 score (`sklearn.metrics`)
- **Dummy Classifier**: Baseline classifier (`sklearn.dummy.DummyClassifier`)
