import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def preprocess_data(file_path):
    """
    Preprocesses the data into a format suitable for FP-growth.
    """
    df = pd.read_csv(file_path)

    # Handle null values (optional, adjust based on your data)
    df.fillna('', inplace=True)

    # Extract the number of items (assuming the first column)
    num_items_per_transaction = df.iloc[:, 0]

    # Extract remaining columns as transactions (assuming items start from second column)
    transactions = df.iloc[:, 1:].values.tolist()

    # Filter out empty transactions
    transactions = [t for t, num_items in zip(transactions, num_items_per_transaction) if num_items > 0]

    # Apply TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    return df_encoded

def get_frequent_items(input_item, rules):
    """
    Finds items frequently bought with the input item.
    """
    # Filter rules for the given product
    filtered_rules = rules[rules['antecedents'].apply(lambda x: input_item in x)]

    # Get consequents
    frequent_items = []
    for _, row in filtered_rules.iterrows():
        consequents = list(row['consequents'])
        frequent_items.extend(consequents)

    # Remove empty items
    frequent_items = [item for item in frequent_items if item]

    # Count item occurrences and sort by frequency
    item_counts = Counter(frequent_items)
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    return [item for item, count in sorted_items[:3]]  # Return top 3 items

def calculate_metrics(input_item, recommended_items, df_encoded, frequent_itemsets):
    """
    Calculates support, confidence, and lift for recommended items.
    """
    total_transactions = len(df_encoded)

    calculated_metrics = []
    for item in recommended_items:
        itemset = {input_item, item}
        support = frequent_itemsets[frequent_itemsets['itemsets'] == itemset]['support'].values[0]
        confidence = support / df_encoded[input_item].sum()
        lift = support / (df_encoded[input_item].sum() * df_encoded[item].sum() / total_transactions)
        calculated_metrics.append((item, support, confidence, lift))

    return calculated_metrics

def apply_kmeans(df_encoded, n_clusters):
    """
    Applies K-Means clustering to the preprocessed transaction data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_encoded)
    cluster_labels = kmeans.labels_
    return kmeans, cluster_labels

def plot_elbow(df_encoded, max_clusters=10):
    """
    Plots the Elbow curve to find the optimal number of clusters.
    """
    inertia = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_encoded)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters+1), inertia, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

def recommend_items(rules):
    """
    Interactively recommends items based on user input.
    """
    def on_button_click(button):
        input_item = text_input.value
        if input_item.lower() == 'exit':
            print("Exiting recommendation system.")
            return
        
        # Validate input
        if input_item not in df_encoded.columns:
            print(f"Product '{input_item}' not found in transactions. Please try another product.")
            return
        
        recommendations = get_frequent_items(input_item, rules)
        if not recommendations:
            print(f"No recommendations found for '{input_item}'.")
        else:
            print(f"Recommended items to buy with '{input_item}': {', '.join(recommendations)}")
    
    # Create a text input and button widget
    text_input = widgets.Text(description="Product:")
    button = widgets.Button(description="Get Recommendations")
    button.on_click(on_button_click)
    
    display(text_input, button)

# Define file path (assuming the file is already uploaded at this path)
file_path = '/content/DATASET.csv'

# Preprocess data
df_encoded = preprocess_data(file_path)

# Apply FP-Growth Algorithm
min_support = 0.001  # Adjust as needed
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

# Generate association rules
min_lift = 1.0  # Adjust as needed
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

# Apply K-Means Clustering
n_clusters = 5  # You can use the elbow method to determine the optimal number
kmeans_model, cluster_labels = apply_kmeans(df_encoded, n_clusters)

# Add the cluster labels to the encoded dataframe
df_encoded['Cluster'] = cluster_labels

# Analyze clusters
for cluster in range(n_clusters):
    print(f"\nCluster {cluster}:")
    cluster_data = df_encoded[df_encoded['Cluster'] == cluster]
    top_items = cluster_data.sum().sort_values(ascending=False).head(10)  # Top 10 most common items in the cluster
    print(top_items)

# Optional: Run the elbow method to visualize optimal clusters
plot_elbow(df_encoded, max_clusters=10)

# Run recommendation system
recommend_items(rules)
