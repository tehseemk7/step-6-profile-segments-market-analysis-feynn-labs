import pandas as pd
import numpy as np
np.random.seed(0)
df = pd.DataFrame({'Feature1': np.random.rand(300),
    'Feature2': np.random.rand(300),
    'Cluster': np.random.randint(0, 4, 300)
})
segment_means = df.groupby('Cluster').mean()
print("Key Characteristics of Market Segments:")
print(segment_means)

##Traditional Approaches to Profiling Market Segments##
segment_summary = df.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
print("Traditional Profiling Summary:")
print(segment_summary)

##Segment Profiling with Visualisations##
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.barplot(x='Cluster', y='Feature1', data=df, estimator=np.mean, ci=None, palette='viridis')
plt.title('Mean of Feature1 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean of Feature1')
plt.show()

# Box plot for Feature1
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Feature1', data=df, palette='viridis')
plt.title('Box Plot of Feature1 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Feature1')
plt.show()

##Assessing Segment Separation
import sys
sys.path.append("C:/Users/tehse/AppData/Roaming/Python/Python312/site-packages")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[['Feature1', 'Feature2']])
# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X_pca)

# Create DataFrame for plotting
plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['Cluster'] = clusters

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=plot_df, palette='viridis', s=100, alpha=0.7)
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Segment Separation Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


##Checklist
import sys
sys.path.append("C:/Users/tehse/AppData/Roaming/Python/Python312/site-packages")
from sklearn.cluster import KMeans
def checklist():
    tasks = [
        "Use the selected segments from Step 5.",
        "Visualise segment profiles to learn about what makes each segment distinct.",
        "Use knock-out criteria to check if any of the segments currently under consideration should be eliminated.",
        "Pass on the remaining segments to Step 7 for describing."
    ]
    print("Step 6 Checklist:")
    tasks = ['task1', 'task2', 'task3']
    task_status = {
    'task1': 'completed',
    'task2': 'pending',
    'task3': 'completed'
}
    for task in tasks:
        response = input(f"Has the task '{task}' been completed? (yes/no): ")
        if response.lower() == 'yes':
            print(f"Task '{task}' completed.")
        else:
            print(f"Task '{task}' is pending.")
            
checklist()

    
            
            
    
     