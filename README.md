Market Basket Analysis and Clustering
This project demonstrates how to perform market basket analysis using the FP-Growth algorithm and K-Means clustering on transaction data. The goal is to identify frequently bought items together and cluster transactions for better insights.

Table of Contents
Installation
Usage
Functions
preprocess_data
get_frequent_items
calculate_metrics
apply_kmeans
plot_elbow
recommend_items
License
Installation
To run this project, you need to have Python installed along with the following libraries:

pip install pandas mlxtend scikit-learn matplotlib ipywidgets

Usage
Preprocess the Data: Load and preprocess the transaction data.
Apply FP-Growth Algorithm: Identify frequent itemsets.
Generate Association Rules: Create rules based on the frequent itemsets.
Apply K-Means Clustering: Cluster the transactions.
Analyze Clusters: Examine the clusters for insights.
Run Recommendation System: Interactively recommend items based on user input.
Functions
preprocess_data
Python

def preprocess_data(file_path):
    """
    Preprocesses the data into a format suitable for FP-growth.
    """
AI-generated code. Review and use carefully. More info on FAQ.
get_frequent_items
Python

def get_frequent_items(input_item, rules):
    """
    Finds items frequently bought with the input item.
    """
AI-generated code. Review and use carefully. More info on FAQ.
calculate_metrics
Python

def calculate_metrics(input_item, recommended_items, df_encoded, frequent_itemsets):
    """
    Calculates support, confidence, and lift for recommended items.
    """
AI-generated code. Review and use carefully. More info on FAQ.
apply_kmeans
Python

def apply_kmeans(df_encoded, n_clusters):
    """
    Applies K-Means clustering to the preprocessed transaction data.
    """
AI-generated code. Review and use carefully. More info on FAQ.
plot_elbow
Python

def plot_elbow(df_encoded, max_clusters=10):
    """
    Plots the Elbow curve to find the optimal number of clusters.
    """
AI-generated code. Review and use carefully. More info on FAQ.
recommend_items
Python

def recommend_items(rules):
    """
    Interactively recommends items based on user input.
    """
AI-generated code. Review and use carefully. More info on FAQ.
License
This project is licensed under the MIT License.
