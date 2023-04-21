# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

diabetes_dataset = pd.read_csv(r'diabetes.csv')
diabetes_dataset.head()

diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# loading the saved model


# creating a function for Prediction

def diabetes_prediction(input_data):
  # changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

  prediction = classifier.predict(input_data_reshaped)
  print(prediction)

  if (prediction[0] == 0):
    return 'The person is not diabetic'
  else:
    return 'The person is diabetic'


def main():
  # giving a title
  st.title('Diabetes Prediction Web App')

  # getting the input data from the user

  Pregnancies = st.text_input('Number of Pregnancies')
  Glucose = st.text_input('Glucose Level')
  BloodPressure = st.text_input('Blood Pressure value')
  SkinThickness = st.text_input('Skin Thickness value')
  Insulin = st.text_input('Insulin Level')
  BMI = st.text_input('BMI value')
  DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
  Age = st.text_input('Age of the Person')

  # code for Prediction
  diagnosis = ''

  # creating a button for Prediction

  if st.button('Diabetes Test Result'):
    diagnosis = diabetes_prediction(
      [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

  st.success(diagnosis)


if __name__ == '__main__':
  main()