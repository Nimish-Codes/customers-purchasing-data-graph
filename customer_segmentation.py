import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    st.title("Customer Segmentation App")
    st.sidebar.header("Parameters")
    
    # Sample customer data (age, income, purchase frequency, average purchase amount)
    data = {
        'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'Income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'Purchase_Frequency': [2, 3, 1, 4, 2, 5, 3, 4, 2, 1],
        'Avg_Purchase_Amount': [50, 60, 40, 70, 50, 80, 60, 70, 50, 40]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Perform k-means clustering
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=5, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = kmeans.fit_predict(scaled_data)

    # Visualize the clusters
    st.write("### Customer Segmentation")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['Income'], df['Avg_Purchase_Amount'], c=df['Segment'], cmap='viridis', s=50, alpha=0.5, label=df['Segment'])
    plt.xlabel('Income')
    plt.ylabel('Avg_Purchase_Amount')
    plt.title('Customer Segmentation')
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Segments")
    st.pyplot(fig)

    # Display segment profiles
    st.write("### Segment Profiles")
    segment_profiles = df.groupby('Segment').mean()
    st.write(segment_profiles)

st.set_option('deprecation.showPyplotGlobalUse', False)

if __name__ == "__main__":
    main()
