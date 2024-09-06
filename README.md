# MLETask

Welcome to the MLETask Package repository! This repository contains a Python package specifically designed to tackle the famous Titanic dataset. The goal of this package is to provide a comprehensive solution for training and evaluating machine learning models on the Titanic dataset, facilitating the exploration of passenger demographics and predicting their survival probabilities.

## About the Titanic Dataset

The Titanic dataset is a well-known and widely used dataset in the field of data analysis and machine learning. It comprises information about passengers who were aboard the RMS Titanic during its ill-fated maiden voyage in 1912. The dataset offers a rich variety of features, including passenger attributes such as age, gender, class, and family relationships. The most crucial feature, "Survived," determines whether a passenger survived the tragic maritime disaster.

## Purpose of the Package

The main purpose of this package is to provide a user-friendly and efficient solution for ML engineers, data scientists to explore, analyze, and build predictive models on the Titanic dataset. The package is built with object-oriented programming principles and adheres to best coding practices, ensuring a robust and maintainable codebase.

## Features and Capabilities

The Titanic ML Package offers the following key features:

- **API For survivors:** A API where by sending the information about the person such as: 'Age', 'Embarked', 'PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin' the model generates a prediction about the survival of that person. In specific the model uses the variables: "Embarked", "Sex", "Pclass", "Age", "SibSp", "Parch", "Fare".

- **Model Training and Evaluation:** The package provides a streamlined workflow for training machine learning models on the Titanic dataset. Users can experiment with different algorithms (random_forest and xgboost), optimize hyperparameters using the optuna library The package also includes evaluation the Accuracy, precision, recall and F1Score metric. The logging of the models is done using Mlflow. 

## Getting Started

To get started with the MLETask Package you just need to clone the repository and run the following code: 

### Usage: 
search for the build_docker.sh and run it in the CMD. Once the code ends its run two docker containers will be deployed one for the model training and another one for the api service. In specific the localhost:8000 endpoint is used fot the API where you can use the localhost:8000/predict endpoint. 
On the localhost:5000 endpoint you should see the mlflow ui that is shown in order to check the models metrics. 
