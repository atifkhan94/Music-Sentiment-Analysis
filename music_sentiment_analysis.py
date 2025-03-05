import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('music_sentiment_dataset.csv')

# Basic data exploration
def explore_data():
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSentiment Distribution:")
    print(df['Sentiment_Label'].value_counts())
    print("\nGenre Distribution:")
    print(df['Genre'].value_counts())

# Visualize sentiment distribution
def plot_sentiment_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Sentiment_Label')
    plt.title('Distribution of Sentiment Labels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()

# Analyze music features by sentiment
def analyze_music_features():
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Sentiment_Label', y='Tempo (BPM)')
    plt.title('Tempo Distribution by Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tempo_by_sentiment.png')
    plt.close()

# Build predictive model
def build_model():
    # Prepare features
    features = ['Tempo (BPM)', 'Genre', 'Mood', 'Energy', 'Danceability']
    target = 'Sentiment_Label'
    
    # Create a copy for modeling
    model_df = df[features + [target]].copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    model_df[target] = le.fit_transform(model_df[target])
    
    # One-hot encode categorical features
    categorical_features = ['Genre', 'Mood', 'Energy', 'Danceability']
    model_df = pd.get_dummies(model_df, columns=categorical_features, drop_first=True)
    
    # Split data
    X = model_df.drop(target, axis=1)
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

def main():
    print("Music Sentiment Analysis")
    print("-" * 50)
    
    # Perform analysis
    explore_data()
    plot_sentiment_distribution()
    analyze_music_features()
    build_model()

if __name__ == "__main__":
    main()