# Combating-Subscriber-Churn-with-Targeted-Marketing

This project revolves around using machine learning techniques to predict subscriber churn and segment subscribers for a video streaming platform, AZ Watch. The goal is to leverage AI-driven solutions to enhance marketing efforts by predicting which subscribers are likely to churn, as well as finding customer segments with similar behaviors. Below is a description of the steps taken in the project:

1. Data Loading and Preprocessing:
The AZWatch_subscribers.csv dataset was loaded into a pandas DataFrame, containing information about subscribers' behavior over the last year.
The columns include:
subscriber_id: Unique identifier for each subscriber.
age_group: Age range of the subscriber (categorical).
engagement_time: Average time spent by the subscriber per session (numerical).
engagement_frequency: Average weekly number of logins (numerical).
subscription_status: Whether the subscriber remained subscribed (target variable for churn prediction).
Preprocessing:
One-Hot Encoding was applied to the categorical column age_group to convert it into multiple binary columns, making it suitable for machine learning algorithms.
The dataset was split into training and testing sets (80% training, 20% testing).
2. Model Training:
Three different machine learning models were trained for churn prediction:

Logistic Regression: A simple, interpretable model was used to predict churn. The model achieved an accuracy score of 92.5%.

Decision Tree Classifier: This model was trained with a maximum depth of 3 and used the Gini impurity criterion to classify churn. It achieved an accuracy score of 92%.

Random Forest Classifier: An ensemble method using 10 decision trees with a maximum depth of 3 was used. This model achieved the highest accuracy score of 93%.

3. Subscriber Segmentation:
To identify customer segments, the age_group feature was removed, and the remaining numerical features (engagement_time and engagement_frequency) were scaled using StandardScaler.

K-Means Clustering was employed to group subscribers into clusters based on their engagement behavior.

Elbow Method: The elbow method was used to determine the optimal number of clusters (k) by plotting the sum of squared errors (SSE) for different values of k. This helped to find that 3 clusters were most appropriate.

Cluster Assignment: Each subscriber was assigned a cluster label (with 3 clusters chosen). The average feature values (engagement time and engagement frequency) for each cluster were calculated.

4. Results:
The analysis of clusters showed different engagement patterns:
The first cluster might represent highly engaged subscribers.
The second cluster could represent low-engagement but loyal users.
The third cluster could represent users with inconsistent usage patterns.
These insights can be used for targeted marketing, where different strategies can be applied to each cluster to improve retention and attract new subscribers.
Conclusion:
The project successfully used machine learning to predict subscriber churn and identify key segments within the subscriber base. With high accuracy achieved through Logistic Regression, Decision Tree, and Random Forest models, the platform can now refine its marketing strategies. The segmentation also offers valuable insights into user behavior, enabling more personalized marketing and retention efforts.







