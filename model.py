from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd
import pickle

#read csv
df = pd.read_csv('final_df.csv')

features = ['year', 'make', 'model' ,'trim', 'body', 'transmission', 'state', 'condition', 'odometer', 'color', 'interior', 'seller', 'season']
target = 'sellingprice'

X = df[features]
y = df[target]

# encode the categorical values
le = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = le.fit_transform(X[column].astype(str))



# Define the parameters for the XGBoost model that we got from the analysis
params = {
    'subsample': 0.8,
    'n_estimators': 500,
    'max_depth': 12,
    'learning_rate': 0.05,
    'colsample_bytree': 1.0
}
# Initialize the XGBoost model with the specified parameters
model = XGBRegressor(**params)

#create pipeline
pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('model', model)
    ])

#fitting the model
pipeline.fit(X, y)


#save model
pickle.dump(pipeline, open('model.pkl', 'wb'))

# Save the column names instead of the entire X DataFrame
with open('input_columns.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)