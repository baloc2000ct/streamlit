

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#get data
df = pd.read_csv('C:/Users/baloc/PycharmProjects/WebApp/venv/diabetes.csv')
#sub header
st.subheader('Data Information: ')
#show datatable frame
st.dataframe(df)
#statistic on data
st.write(df.describe())
#Data chart
st.bar_chart(df)

#Split the data into independent 'A' and dependent 'Y' variables

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#split the data into 75% Training and 25% testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#get the feature from the user

def get_user_input():

    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 4)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 199, 117)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DFF', 0.0, 67.1, 32.0)
    age = st.sidebar.slider('age', 21, 81, 29)

    #storde dictionary
    user_data = {'pregnancies': pregnancies, 'glucose': glucose,
                 'blood_pressure': blood_pressure, 'skin_thickness': skin_thickness,
                 'insulin': insulin, 'BMI':BMI,
                 'DPF': DPF, 'age': age
                 }

    #modify the data into the data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the data user
user_input = get_user_input()

#Subheader and display users input

st.subheader('User Input:')
st.write(user_input)

#Train the model

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

#store the models predictions
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification:')
st.write(prediction)

if(prediction[0]==0):

    st.write("The person is not diabetes")
else:

    st.write("The person is diabetes")


