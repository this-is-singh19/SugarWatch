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
st.title('SugarWatch - Diabetic Prediction Web App')
st.sidebar.header('Patient Input Parameters')
st.subheader('Training Data Set Information')
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
st.title('Visualized Patient Report')

# COLOR FUNCTION
if user_result[0] == 0:
    color = 'blue'
else:
    color = 'red'

# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='coolwarm', ax=ax1)
sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color, marker='s', ax=ax1)
ax1.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='viridis', ax=ax2)
sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color, marker='s', ax=ax2)
ax2.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='rocket', ax=ax3)
sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color, marker='s', ax=ax3)
ax3.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st, ax4 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues', ax=ax4)
sns.scatterplot(x=user_data['Age'], y=user_data['SkinThickness'], s=150, color=color, marker='s', ax=ax4)
ax4.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 110, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i, ax5 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket', ax=ax5)
sns.scatterplot(x=user_data['Age'], y=user_data['Insulin'], s=150, color=color, marker='s', ax=ax5)
ax5.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 900, 50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi, ax6 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow', ax=ax6)
sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color, marker='s', ax=ax6)
ax6.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 70, 5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf, ax7 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr', ax=ax7)
sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color=color, marker='s', ax=ax7)
ax7.grid(True)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 3, 0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)

# OUTPUT
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
st.subheader('Accuracy:')
st.header(f"{round(accuracy_score(y_test, gbm.predict(x_test))*100, 2)}%")