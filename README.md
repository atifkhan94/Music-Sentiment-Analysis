# Music Sentiment Analysis

A Python-based project that analyzes sentiment in music using various features like tempo, genre, mood, energy, and danceability. The project implements machine learning techniques to predict sentiment labels and provides insightful visualizations of music characteristics.

## Features

- **Data Exploration**: Basic analysis of music dataset including shape, columns, and distribution of sentiments and genres
- **Sentiment Visualization**: Visual representation of sentiment distribution across the dataset
- **Music Feature Analysis**: Analysis of musical features (e.g., Tempo) and their relationship with sentiment
- **Predictive Modeling**: Implementation of Random Forest Classifier for sentiment prediction
- **Performance Metrics**: Detailed classification reports and model evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd music-sentiment-analysis
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Ensure your music dataset (`music_sentiment_dataset.csv`) is in the project directory

2. Run the main script:
```bash
python music_sentiment_analysis.py
```

3. The script will:
   - Display basic data exploration results
   - Generate visualization plots (saved as PNG files)
   - Build and evaluate the sentiment prediction model

## Project Structure

- `music_sentiment_analysis.py`: Main Python script containing the analysis pipeline
- `music_sentiment_dataset.csv`: Dataset containing music features and sentiment labels
- `sentiment_distribution.png`: Generated visualization of sentiment distribution
- `tempo_by_sentiment.png`: Generated visualization of tempo distribution by sentiment

## Model Features

The sentiment prediction model uses the following features:
- Tempo (BPM)
- Genre
- Mood
- Energy
- Danceability

## Output

The script generates:
1. Data exploration statistics
2. Visualization plots:
   - Sentiment distribution
   - Tempo distribution by sentiment
3. Model performance metrics:
   - Classification report
   - Prediction accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors and users of this project
- Inspired by the need for understanding emotional aspects of music