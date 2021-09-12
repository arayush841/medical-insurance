from os import write
from matplotlib.pyplot import step
import streamlit as st
import numpy as np
import sklearn
import pickle   

st.set_page_config(page_title='Medical Insurance Cost Prediction', page_icon="🏥")

@st.cache
def load_model(file):
    model = pickle.load(open(file, "rb"))
    return model

mymodel = load_model('model_new.pkl')

st.title('Medical Insurance Cost Prediction')
#st.write('Summer Internship Project by Ayush Raj, B.Tech(CSE)-5th Semester')

age = st.number_input("Enter your Age.", min_value = 18, max_value= 100, step = 1)
sex = st.selectbox('Enter your Gender. ( Male : 1 , Female : 0 )', (0, 1))
bmi = st.number_input("Enter your BMI. ( Body Mass Index ( BMI = weight(kg) / height*height(m^2) )", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Do you have any childrens? If yes tell us the no. of children you have else select 0", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Are you a smoker? ( Yes : 1 , No : 0)", (0,1))
region = st.selectbox("Select your region. ( Southeast : 0 , Southwest : 1, Northeast : 2, Northwest : 3)", (0,1,2,3))

prediction = mymodel.predict([[age,sex,bmi,children,smoker,region]])

cost = np.round(prediction[0], 2)

cost_in_inr = cost*73

status = st.button("Click to check your expected Medical Insurance Cost")
if status:
    if (cost<0 and cost_in_inr<0):
        st.write("Sorry! We don't have much data which matches your entries for predicting the cost of medical insurance.")
    else:
        st.write("Predicted Medical Insurance Cost for you in USD is : " + str(cost))
        st.write("Predicted Medical Insurance Cost for you in INR is : " + str(round(cost_in_inr,2)))
