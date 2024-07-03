
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import  mean_squared_error

# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams["figure.figsize"] = [10,5]
# # Ignore warnings
# import warnings
# # Set the warning filter to ignore FutureWarning
# warnings.simplefilter(action = "ignore", category = FutureWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Cancer Survival Prediction App")
# upload dataset
uploaded_file=st.file_uploader("upload  your dataset",type=["xlsx","csv"])
# if uploaded_file is not None:
df=pd.read_excel(uploaded_file)


st.write(df.head(5))

st.write(df.shape)
# display column names with their datatypes
st.write("column names and their datatypes",df.dtypes)

st.write("### Null Values")
st.write(df.isnull().sum())

# df.dropna(inplace=True)
# st.write("###after dropping Null Values")
# st.write(df.isnull().sum())

st.header("Data Visualization")

st.write("#### Number of people survived")
st.write(df['survived'].value_counts())
# st.write(sns.countplot(x="survived",data=df,color="green"))
fig, ax = plt.subplots()
sns.countplot(x="survived", data=df, ax=ax, color="green")
st.pyplot(fig)

st.write("### Gender Distribution")
st.write(df['gender'].value_counts())
# st.write(sns.countplot(x="gender",data=df))
fig, ax = plt.subplots()
sns.countplot(x="gender", data=df, ax=ax, color="blue")
st.pyplot(fig)


# Group by cancer_stage and count the number of survivors and non-survivors

fig, ax = plt.subplots()
sns.countplot(x="cancer_stage", data=df, ax=ax, hue="survived")
st.write(plt.title("survival by stages"))
st.write(plt.xlabel("cancer_stage"))
st.write(plt.ylabel("count"))
st.pyplot(fig)

# survival by smoking status
survival_by_smoking=df.groupby(["smoking_status","survived"]).size().unstack(fill_value=0)
st.write((survival_by_smoking))

fig, ax = plt.subplots()
sns.countplot(x="smoking_status", data=df, ax=ax, hue="survived")
plt.title('Survival by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
st.pyplot(fig)

st.write("#### survival by gender")
fig, ax = plt.subplots()
sns.countplot(data=df, x='gender',ax=ax, hue='survived')
plt.title('Survival by gender')
plt.xlabel('gender')
plt.ylabel('Count')
st.pyplot(fig)

st.write("#### survival rate by  asthma")
fig,ax=plt.subplots()
sns.countplot(data=df, x='asthma',ax=ax, hue='survived')
st.pyplot(fig)

st.write("max age is:",df["age"].max())
st.write("min age is ",df["age"].min())

age_bins = [18, 30, 45, 60, 75, 95]
age_labels = ['18-30', '31-45', '46-60', '61-75', '76-95']
df['age_category'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
st.bar_chart(df["age_category"].value_counts())

st.write("#### servival by age category")
fig,ax=plt.subplots()
sns.countplot(data=df, x='age_category', ax=ax, hue='survived')
st.pyplot(fig)

st.write("#### servival by family history")
fig,ax=plt.subplots()
sns.countplot(data=df, x='family_history',ax=ax, hue='survived')
st.pyplot(fig)

st.write("#### servival as by treatment type")
fig,ax=plt.subplots()
sns.countplot(data=df, x='treatment_type', hue='survived')
st.pyplot(fig)

fig,ax=plt.subplots()
sns.countplot(data=df, x='hypertension', hue='survived')
st.pyplot(fig)


fig,ax=plt.subplots()
sns.boxplot(data=df, x="cholesterol_level", ax=ax)
st.pyplot(fig)

st.write("### heatmap")
# Filter the DataFrame to include only numeric columns
numeric_df = df.select_dtypes(include=[float, int])
# Calculate the correlation matrix
correlation = numeric_df.corr()
# Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(correlation, cmap="BrBG", annot=True, ax=ax)
# Display the plot in the Streamlit app
st.pyplot(fig)


# feature Engineering
st.header(" Feature Engineering")
df=df.drop(["country","cancer_stage","family_history","smoking_status","treatment_type",
            "diagnosis_date","beginning_of_treatment_date","end_treatment_date","gender","age_category",],axis=1)
st.write("#### data after dropping columns")
st.write(df.head())
st.write(df.shape)

x=df.drop("survived",axis=1)
st.write("input features",x.shape)
y=df["survived"]
st.write("target variable",y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=101)
st.subheader("Train-Test Split")
st.write("X_train shape:")
st.write(x_train.shape)
st.write("X_test shape:")
st.write(x_test.shape)
st.write("y_train shape:")
st.write(y_train.shape)
st.write("y_test shape:")
st.write(y_test.shape)

# Model selection
st.write("## Model Training")
model_options = ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "Gradient Boosting Regressor", "SVR"]
selected_model = st.selectbox("Select a model", model_options)

if selected_model == "Linear Regression":
        model = LinearRegression()
elif selected_model == "Random Forest Regressor":
        model = RandomForestRegressor()
elif selected_model == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
elif selected_model == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
elif selected_model == "SVR":
        model = SVR()

# Train the selected model
model.fit(x_train, y_train)
predictions = model.predict(x_test)

mse = mean_squared_error(y_test, predictions)
st.write("### Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")