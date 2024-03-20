# Car Price Prediction Analysis

## 1. Objective
The goal of this project is to develop a model capable of predicting car prices using various car attributes. Through examining the relationship between car features and their prices, our aim is to construct a linear regression model that accurately estimates car prices, serving as a valuable resource for both buyers and sellers in the automotive market.

## 2. Overview
The ability to predict car prices accurately is crucial for various market participants, including dealerships, manufacturers, and consumers. This project leverages a dataset with detailed car attributes to construct a model that elucidates the factors affecting car prices.

## 3. Dataset Insights
The dataset encompasses diverse information about cars, detailed as follows:

| Attribute          | Description                                                                |
|--------------------|----------------------------------------------------------------------------|
| ID                 | Unique identifier for each car.                                            |
| Symboling          | Assigned rating indicating the car's risk level.                           |
| Name               | Car model name.                                                            |
| Fuel Types         | Type of fuel used by the car.                                              |
| Aspiration         | Engine aspiration method (standard or turbocharged).                       |
| Door Numbers       | Number of doors on the car.                                                |
| Car Body           | Style of the car body.                                                     |
| Drive Wheels       | Type of drive wheel configuration.                                         |
| Engine Location    | Placement of the engine within the car.                                    |
| Wheelbase          | Distance between front and rear wheels.                                    |
| Engine Size        | Engine displacement volume.                                                |
| Fuel System        | Type of fuel delivery system.                                              |
| Bore Ratio         | Diameter of engine cylinders.                                              |
| Stroke             | Length of the piston stroke.                                               |
| Compression Ratio  | Ratio of combustion chamber volume at different piston positions.          |
| Horsepower         | Engine power output.                                                       |
| Peak RPM           | Maximum engine revolutions per minute.                                     |
| City MPG           | Estimated city driving fuel efficiency.                                    |
| Highway MPG        | Estimated highway driving fuel efficiency.                                 |
| Price              | Car price.                                                                 |

## 4. Methodology
The process involves several key steps, from data preparation through to model evaluation:

### Data Loading and Inspection
- Load the dataset and inspect its structure, examining both features and the target variable.
  
### Data Preprocessing
- Address missing values, encode categorical variables, and scale features as necessary.

### Exploratory Data Analysis (EDA)
- Investigate the dataset to find correlations between features and the target variable, and visualize these relationships.

### Feature Selection
- Identify and select the features that significantly correlate with the target variable, price, for inclusion in the model.

### Model Training
- Split the dataset into training and test sets and train a linear regression model using the selected features.

### Model Evaluation
- Utilize metrics like MSE, MAE, R2, and RMSE to evaluate the model's accuracy and performance.

### Visualization
- Create visualizations to compare actual vs. predicted prices and to illustrate the regression model's effectiveness.

## 5. Detailed Model Training Procedure

### Importing Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```
### Loading Data:
Reads the data from a CSV file into a pandas DataFrame.
```python
df = pd.read_csv('car_dataset.csv')
```
### EDA:
Inspects the data, summarizes statistics, explores unique values, and identifies columns.
```python
# check the shape
df.shape
# first five rows of the dataframe
df.head()
# describe the dataframe with some statistical info
df.describe()
# check data types in the dataframe
df.info()
# check unique data for each feature in the dataframe
df.nunique()
# column names of the dataframe
df.columns
```
### Data Selection:
Selects numerical columns for analysis.
```python
numerical_columns = ['wheelbase', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg']
```
### Correlation Analysis:
Calculates correlation coefficients between numerical features and the target variable ('price').
```python
# Calculate correlation coefficients with respect to "price"
correlation_with_price = df[numerical_columns].corrwith(df['price']).abs().sort_values(ascending=False)
```
### Creating a New DataFrame:
Creates a new DataFrame using 'enginesize' and 'price' for model training.
```python
x = new_df['enginesize']
y = new_df['price']
```
### Comparing Engine Size and Fuel Type with Price using Scatter Plot
<img src="picture/2.1.png" width="500" alt="">
<img src="picture/2.2.png" width="500" alt="">

### Splitting Data:
Splits the data into training and testing sets.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
### Model Creation:
Initializes a Linear Regression model.
```python
model = LinearRegression()
```
### Model Training:
Fits the model to the training data.
```python
model.fit(x_train.values.reshape(-1,1), y_train)
```
### Evaluation metric - R2
Calculates evaluation metric - R2 score.
### Plotting Actual vs. Predicted Values:
Visualizes the relationship between actual and predicted prices.

Plotting Regression Model Line:
Plots the regression line to visualize the model's fit.

<img src="picture/2.3.png" width="500" alt="">

Predicting Prices based on Engine Size:
Generates predicted prices based on engine size.

<img src="picture/2.4.png" width="500" alt="">

### Predicting Prices based on Engine Size:
Generates predicted prices based on engine size.
| Engine Size | Predicted Price |
|-------------|-----------------|
| 91          | 7339.34         |
| 161         | 18841.42        |
| 136         | 14733.53        |
| 61          | 2409.87         |
| 109         | 10297.01        |
| 146         | 16376.69        |
| 92          | 7503.65         |
| 92          | 7503.65         |
| 181         | 22127.73        |
| 92          | 7503.65         |
| 164         | 19334.36        |
| 203         | 25742.67        |
| 70          | 3888.71         |
| 134         | 14404.90        |
| 90          | 7175.02         |
| 146         | 16376.69        |
| 132         | 14076.27        |
| 136         | 14733.53        |
| 110         | 10461.33        |
| 92          | 7503.65         |
| 110         | 10461.33        |
| 120         | 12104.48        |
| 132         | 14076.27        |
| 146         | 16376.69        |
| 171         | 20484.57        |
| 97          | 8325.23         |
| 98          | 8489.54         |
| 120         | 12104.48        |
| 98          | 8489.54         |
| 97          | 8325.23         |
| 109         | 10297.01        |
| 109         | 10297.01        |
| 151         | 17198.26        |
| 122         | 12433.11        |
| 97          | 8325.23         |
| 209         | 26728.56        |
| 109         | 10297.01        |
| 121         | 12268.80        |
| 90          | 7175.02         |
| 304         | 42338.53        |
| 90          | 7175.02         |

## 6. Insights and Model Efficacy
The comprehensive analysis and deployment of a linear regression model have culminated in a predictive framework adept at forecasting car prices with engine size as the primary variable. Key takeaways from this endeavor include:

**Critical Findings:**

- **Significance of Engine Size:** The analysis confirms that engine size is paramount in predicting car prices. It registers the highest correlation with the car's price over other assessed numerical features.
- **Model Efficacy:** Demonstrating an R-squared value close to 0.78, the linear regression model achieves commendable predictive performance. This metric indicates that the model accounts for a significant proportion of the price variability, showcasing its predictive precision.

- **Predictive Insights:** Despite the natural price variations in the automotive market, the model closely approximates real-world prices. The accuracy of the model, as supported by both evaluation metrics and visual plots, underscores its utility in forecasting prices based on engine size, aiding stakeholders in informed decision-making.

The diligent analytical and modeling processes have yielded a dependable framework for accurately forecasting car prices, with engine size identified as a crucial factor.

## 7. Project Wrap-up
This initiative has successfully forged a predictive model that emphasizes the pivotal role of engine size in determining car prices. This revelation underscores engine size as a crucial consideration for both purchasers and vendors in the automotive sector.

Looking ahead, there's potential for further refinement and enhancement of the model to elevate its accuracy and extend its relevance. Introducing more features and applying sophisticated modeling techniques could unravel additional layers of the intricate car pricing landscape.

In essence, the developed predictive model stands as a strategic asset for the automotive industry's stakeholders, offering actionable insights to guide pricing strategies, support decision-making, and ultimately, amplify operational efficiency and market effectiveness.





