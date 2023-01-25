import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('./heart.csv')

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach']

X = dataset[features]
Y = dataset['target']

model = LogisticRegression()
model.fit(X, Y)




st.header("""
    Heart Disease Prediction App

    This app predicts whether you have a **Heart Disease** or not ! 

    Data obtained from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
""")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input('**Age**', min_value=18, max_value=100, value=20 )
    cp = st.number_input('**Chest pain type**', min_value=0, max_value=3, value=1)

with col2:
    sex = st.selectbox('**Gender**',('Male', 'Female'))
    chol = st.number_input('**serum cholestoral in ***mg/dl***** ', min_value=100, max_value=300, value=200)

with st.container():
    trestbps = st.number_input('**resting blood pressure ***(in mm Hg on admission to the hospital)*****', min_value=50, max_value=250, value=100)
    fbs = st.selectbox('**fasting blood sugar**', ('True', 'False'))

with st.container():
    thalach = st.number_input('**maximum heart rate achieved** ', min_value=60, max_value=300, value=120)


if sex == 'Male':
    sex = 1
else:
    sex = 0    

if fbs == 'True':
    fbs = 1 
else:
    fbs = 0    


data = {'age':[age], 'sex':[sex], 'cp':[cp], 'trestbps':[trestbps],  'chol':[chol], 'fbs':[fbs], 'thalach':[thalach]}
df = pd.DataFrame(data)


if st.button('predict'):
    prediction = model.predict(df) 

    if prediction == 1:
        st.write('You have a **Heart Disease**.')

    else:
        st.write('Your heart is in a **good state**.')


    









