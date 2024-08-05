# Megaline Prediction Model - Subscribers Behavior Analysis

**Data Analyst Bootcamp** @ Tripleten<br>
Sprint 11 Project - Intro to Machine Learning

## Context

Megaline, a leading mobile carrier, strives to enhance customer satisfaction by ensuring subscribers are on the most suitable plan for their needs. Despite the introduction of newer plans, Smart and Ultra, many customers continue to use legacy plans. To address this, Megaline has launched a program to transition customers to the newer, optimized plans. The Customer Insights Department aims to develop a predictive model to recommend either the Smart or Ultra plan based on user behavior.

## Overview
We developed a classification model to predict the most appropriate plan for each user, using monthly behavior data from subscribers who have already switched to the new plans. The dataset includes metrics such as the number of calls made, call duration, number of text messages sent, internet traffic used, and the plan for the current month.

The project involved splitting the data into training, validation, and test sets, experimenting with various models and hyperparameters, and evaluating the models' performance. We set a minimum accuracy threshold of 75% to ensure the model's effectiveness. Finally, a sanity check was conducted to verify the model's robustness. The ultimate goal is to deliver a high-accuracy model that facilitates personalized plan recommendations, improving customer satisfaction and loyalty.

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
