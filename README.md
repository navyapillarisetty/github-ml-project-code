# github-ml-project-code
The dataset used for this project is Energy Dataset from Kaggle (https://www.kaggle.com/datasets/shub218/energy-data-1990-2020?resource=download)
The dataset contains information regarding the energy consumed and produced by each country from years 1990 to 2020 i.e. 30 years. The energy profile of 44 countries is given in the dataset. The following work is performed on the dataset:
1. Dividing the dataset into groups based on the "Country" variable.
2. Descriptive analysis is performed on each variable for each group.
3. Feature engineering and feature scaling is performed for each group.
The main objective of this project is to use different forecasting models for each group and forecast "Surplus Energy Production" and identify the better model which could forecast with fewer errors for this small range of data in each group.
In this project, I have used different models like Recurrent Neural Networks with GRU, Recurrent Neural Networks with LSTM, SimpleRNN, Convolutional Neural Networks, Feedforward Neural Networks, XGBoost Regressor, Linear Regression Model, and ARIMA on all 44 groups within the Energy dataset.
The below figure shows the Mean Absolute Error of each model and it shows that Linear regression works well with each country.

![image](https://github.com/navyapillarisetty/github-ml-project-code/assets/130261532/699c7802-e837-4ccf-9375-af97dbaed04b)

