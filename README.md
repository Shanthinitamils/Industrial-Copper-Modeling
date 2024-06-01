# Industrial-Copper-Modeling




# Introduction
Enhance your proficiency in data analysis and machine learning with our "Industrial Copper Modeling" project. In the copper industry, dealing with complex sales and pricing data can be challenging. Our solution employs advanced machine learning techniques to address these challenges, offering regression models for precise pricing predictions and lead classification for better customer targeting. You'll also gain experience in data preprocessing, feature engineering, and web application development using Streamlit, equipping you to solve real-world problems in manufacturing.


## Key technology and skill

-Python
-Numpy
-Pandas
-Scikit-Learn
-Matplotlib
-Seaborn
-Pickle
-Streamlit
## Installation
To run this project pip Installation
pip install numpy
pip install pandas
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
pip install streamlit
    
## Run Locally



Go to the project directory

https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit?usp=sharing&ouid=104970222914596366601&rtpof=true&sd=true
```

```


## Running Tests

To run tests, run the following command
streamlit run File Name

## Featues

## Data Processing

-Data Understanding: Before diving into modeling, it's crucial to gain a deep understanding of your dataset. Start by identifying the types of variables within it, distinguishing between continuous and categorical variables, and examining their distributions. In our dataset, there might be some unwanted values in the 'Material_Ref' feature that start with '00000.' These values should be converted to null for better data integrity.

-Handling null values:Replace all non-positive values in 'quantity tons', 'selling_price', and 'delivery_date_dif' with np.nan.also filling all missing columns with fillna median values values of that columns

-Encoding and Data processing: We encoding the status and Item type column (categorical value) to numeric using encoding technique

-Skewness - Feature Scaling: Skewness is a common challenge in datasets. Identifying skewness in the data is essential, and appropriate data transformations must be applied to mitigate it. One widely-used method is the log transformation, which is particularly effective in addressing high skewness in continuous variables. This transformation helps achieve a more balanced and normally-distributed dataset, which is often a prerequisite for many machine learning algorithms.

Outliers Handling: Outliers can significantly impact model performance. We tackle outliers in our data by using the Interquartile Range (IQR) method. This method involves identifying data points that fall outside the IQR boundaries and then converting them to values that are more in line with the rest of the data. This step aids in producing a more robust and accurate model.

## Exploratory Data Analysis (EDA) and Feature Engineering:
Skewness visualization:Depending on the findings from the plots, we might need to consider data transformation techniques to address potential skewness and improve the effectiveness of machine learning models trained on this data.
outliers visualization:Outliers are data points that fall significantly outside the overall pattern of the data. While they can sometimes be indicative of errors or unusual events, they can also be valid data points. The decision of how to handle outliers depends on your specific analysis and domain knowledge.
Feature Improvement:The outlier function might not be suitable for the 'width' column if it contains categorical data (e.g., width categories). You might need to consider alternative methods for identifying anomalies in categorical variables, such as looking for categories with very low or high frequencies.

By incorporating these improvements, you can create a more robust and informative outlier detection process for your DataFrame.

## Classification

-Algorithm Assesment:The results dictionary is iterated over, and the algorithm with the highest accuracy score is identified. This is considered the "best" algorithm based on this metric.
It's important to note that accuracy might not always be the best metric depending on the specific problem. Choosing the best metric depends on the cost of false positives and false negatives in your application.

-Algorith Selection:
Although both Randomtreeclassifier(97.4) and Extratressclassifier(97.6) i choose Extratreesclassifier.
-Hyperparameter Tuning with GridSearchCV : To fine-tune our model and mitigate overfitting, we employ GridSearchCV  for hyperparameter tuning. This function allows us to systematically explore multiple parameter values and return the optimal set of parameters. {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 2,n_estimator:50}

-Success and Failure Classification: In our predictive journey, we utilize the 'status' variable, defining 'Won' as Success and 'Lost' as Failure. Data points with status values other than 'Won' and 'Lost' are excluded from our dataset to focus on the core classification task.

-Model Persistence: We conclude this phase by saving our well-trained model to a pickle file. This enables us to effortlessly load the model and make predictions on the status whenever needed, streamlining future applications.
## Regression
-Algorithm Assesment:The results dictionary is iterated over, and the algorithm with the highest accuracy score is identified. This is considered the "best" algorithm based on this metric.
It's important to note that accuracy might not always be the best metric depending on the specific problem. Choosing the best metric depends on the cost of false positives and false negatives in your application.

-Algorith Selection:
Although both Randomtreeclassifier(91.6) and Extratressclassifier(91.2) i choose Randomtreeclassifier.
-Hyperparameter Tuning with GridSearchCV : To fine-tune our model and mitigate overfitting, we employ GridSearchCV  for hyperparameter tuning. This function allows us to systematically explore multiple parameter values and return the optimal set of parameters. {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
-Model Persistence: We conclude this phase by saving our well-trained model to a pickle file. This enables us to effortlessly load the model and make predictions on the status whenever needed, streamlining future applications.
## Lessons Learned

Data is Crucial:

The quality and quantity of data significantly impact model performance. Ensure your data is accurate, comprehensive, and relevant to the prediction task (e.g., predicting copper prices).
Data cleaning and preprocessing are essential steps to prepare the data for modeling. Missing values, outliers, and inconsistencies can negatively affect model training.
Understanding the Domain:

Knowledge of the copper industry and factors affecting copper prices is valuable. This knowledge can help select relevant features for modeling and interpret the results meaningfully.
Choosing the Right Techniques:

Different machine learning algorithms are suitable for various tasks. Understanding the strengths and limitations of algorithms like regression and classification is crucial for selecting the best approach for your specific prediction goal.
Techniques like feature scaling, dimensionality reduction, and hyperparameter tuning can significantly improve model performance.
Evaluation Matters:

Evaluating model performance using appropriate metrics is essential. Accuracy might not always be the best metric, and the choice depends on the cost of false positives and false negatives in your application.
Consider using cross-validation techniques to obtain a more robust estimate of model generalizability.
Dealing with Skewed Data:

Industrial data, like copper prices, might exhibit skewness (uneven distribution). Techniques like log-transformation or using appropriate algorithms that are robust to skewed data can be helpful.
Importance of Feature Engineering:

Feature engineering involves creating new features from existing ones to improve model performance. This can involve combining features, extracting domain-specific insights, or transforming features to enhance their predictive power.
Addressing Imbalance:

If your data is imbalanced (e.g., more data points for lower copper prices), techniques like SMOTE (Synthetic Minority Oversampling Technique) can be used to address this issue and improve model performance for the minority class.
Model Interpretation:

Once you have a well-performing model, understanding the factors that influence its predictions is crucial. Techniques like feature importance analysis can help you identify the most significant features for copper price prediction.
Limitations and Challenges:

Machine learning models are not perfect and can make mistakes. Understanding the limitations of your model and the potential for errors is important for responsible use.
External factors beyond the scope of your model can still impact copper prices. It's important to acknowledge these limitations and incorporate domain expertise when interpreting model predictions.
By reflecting on these potential lessons learned from industrial copper modeling, you can gain valuable insights into real-world machine learning applications and best practices.


## Demo

https://www.linkedin.com/posts/shanthini-tamilselvan-2a0a102b6_coppermodeling-industrialengineering-dataanalysis-activity-7202711565098246144-swtP?utm_source=share&utm_medium=member_desktop

