
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "Indian_Kids_Screen_Time.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ankushpanday2/indian-kids-screentime-2025",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

# print(df)

#Data separation as x and y
y=df[['Avg_Daily_Screen_Time_hr']]
x=df[["Age"]]



#Splitting into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

#Linear regression model
lr=LinearRegression()
lr.fit(x_train,y_train)

#Applying the model to make a prediction
y_lr_train_pred=lr.predict(x_train)
y_lr_test_pred=lr.predict(x_test)

print(y_lr_train_pred)
print("*"*100)
print(y_lr_test_pred)

#Evaluating model performance
train_mse=mean_squared_error(y_train,y_lr_train_pred)
train_r2=r2_score(y_train,y_lr_train_pred)

test_mse=mean_squared_error(y_test,y_lr_test_pred)
test_r2=r2_score(y_test,y_lr_test_pred)

lr_results=pd.DataFrame(["Linear Regression ",train_mse,train_r2,test_mse,test_r2]).transpose()
lr_results.columns=["Model Type","Training data MSE","Training data R2","Testing data MSE","Testing data R2"]

print(lr_results)