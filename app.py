import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('SugarWatch - Diabetic Prediction Web App :syringe:')
st.sidebar.header('Patient Input Parameters')
st.subheader('Training Data Set Information :clipboard:')
st.dataframe(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.dataframe(user_data)

# MODEL
gbm = GradientBoostingClassifier()
gbm.fit(x_train, y_train)
user_result = gbm.predict(user_data)

# VISUALIZATIONS
st.title('Visualized Patient Report :bar_chart:')

# Color function
color = 'blue' if user_result[0] == 0 else 'red'

# Age vs Pregnancies
st.header('Pregnancy Count Comparison')
fig_preg, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='coolwarm', ax=ax1)
sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color, marker='s', ax=ax1)
ax1.set_xlabel('Age')
ax1.set_ylabel('Pregnancies')
ax1.set_title('Comparison of Pregnancy Count')
st.pyplot(fig_preg)

# Age vs Glucose
st.header('Glucose Value Comparison')
fig_glucose, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='viridis', ax=ax2)
sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color, marker='s', ax=ax2)
ax2.set_xlabel('Age')
ax2.set_ylabel('Glucose')
ax2.set_title('Comparison of Glucose Value')
st.pyplot(fig_glucose)

# Age vs Blood Pressure
st.header('Blood Pressure Comparison')
fig_bp, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='rocket', ax=ax3)
sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color, marker='s', ax=ax3)
ax3.set_xlabel('Age')
ax3.set_ylabel('Blood Pressure')
ax3.set_title('Comparison of Blood Pressure')
st.pyplot(fig_bp)

# Age vs Skin Thickness
st.header('Skin Thickness Comparison')
fig_st, ax4 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues', ax=ax4)
sns.scatterplot(x=user_data['Age'], y=user_data['SkinThickness'], s=150, color=color, marker='s', ax=ax4)
ax4.set_xlabel('Age')
ax4.set_ylabel('Skin Thickness')
ax4.set_title('Comparison of Skin Thickness')
st.pyplot(fig_st)

# Age vs Insulin
st.header('Insulin Value Comparison')
fig_i, ax5 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket', ax=ax5)
sns.scatterplot(x=user_data['Age'], y=user_data['Insulin'], s=150, color=color, marker='s', ax=ax5)
ax5.set_xlabel('Age')
ax5.set_ylabel('Insulin')
ax5.set_title('Comparison of Insulin Value')
st.pyplot(fig_i)

# Age vs BMI
st.header('BMI Value Comparison')
fig_bmi, ax6 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow', ax=ax6)
sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color, marker='s', ax=ax6)
ax6.set_xlabel('Age')
ax6.set_ylabel('BMI')
ax6.set_title('Comparison of BMI Value')
st.pyplot(fig_bmi)

# Age vs Diabetes Pedigree Function
st.header('Diabetes Pedigree Function Comparison')
fig_dpf, ax7 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr', ax=ax7)
sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color=color, marker='s', ax=ax7)
ax7.set_xlabel('Age')
ax7.set_ylabel('Diabetes Pedigree Function')
ax7.set_title('Comparison of Diabetes Pedigree Function')
st.pyplot(fig_dpf)

# OUTPUT
st.subheader('Your Report :clipboard:')
output = 'You are not diabetic' if user_result[0] == 0 else 'You are diabetic'
st.title(output)
st.subheader('Accuracy :chart_with_upwards_trend:')
accuracy = accuracy_score(y_test, gbm.predict(x_test)) * 100
st.header(f"{accuracy:.2f}%")
