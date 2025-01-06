import plotly.express as px
import pandas as pd
import json
import argparse

def load_data(file_path):
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.json_normalize(data)

def wrap_text(text, max_words=15):
    if isinstance(text, list):
        text = ' '.join(text)  # Convert list to a single string
    elif isinstance(text, str):
        words = text.split()
        wrapped_lines = []
        for i in range(0, len(words), max_words):
            wrapped_lines.append(' '.join(words[i:i + max_words]))
        return '<br>'.join(wrapped_lines)
    else:
        return str(text)

def create_plot(df, file_path):
    # Preprocess the metadata fields to wrap text
    df['analysis.transcription'] = df['analysis.transcription'].apply(wrap_text)
    df['analysis.analysis.reasoning'] = df['analysis.analysis.reasoning'].apply(wrap_text)
    df['analysis.analysis.recommendations'] = df['analysis.analysis.recommendations'].apply(lambda x: wrap_text(' '.join(x) if isinstance(x, list) else x))

    # Fill missing sentiment scores with the previous value
    df['analysis.analysis.sentiment_score'] = df['analysis.analysis.sentiment_score'].fillna(method='ffill')

    # Calculate the average sentiment score
    average_sentiment = calculate_average_sentiment(df)

    # Create the interactive plot using Plotly
    fig = px.scatter(
        df,
        x="timestamp",
        y="analysis.analysis.sentiment_score",
        color="analysis.analysis.sentiment_score",
        color_continuous_scale="Viridis",
        hover_data={
            "timestamp": True,
            "analysis.analysis.sentiment_score": True,
            "analysis.transcription": True,
            "analysis.analysis.reasoning": True,
            "analysis.analysis.recommendations": True
        },
        title=f"Classroom Sentiment Analysis Over Time (Scaled -5 to 5) - {file_path.split('/')[-2]}"
    )

    # Customize hover template for better readability
    fig.update_traces(
        mode='lines+markers',
        customdata=df[['analysis.transcription', 'analysis.analysis.reasoning', 'analysis.analysis.recommendations']].values,
        hovertemplate=(
            "<b>Timestamp:</b> %{x}<br>"
            "<b>Sentiment Score:</b> %{y}<br>"
            "<b>Transcription:</b> %{customdata[0]}<br>"
            "<b>Reasoning:</b> %{customdata[1]}<br>"
            "<b>Recommendations:</b> %{customdata[2]}<br>"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            font_size=12,  # Set font size
            font_family="Arial",  # Set font family
            align="left",  # Align text to the left
            bgcolor="white",  # Set background color
            bordercolor="black",  # Set border color
            namelength=-1  # Show full name without truncation
        )
    )

    # Add annotation for average sentiment score at the top-left corner
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1, y=1.1,  # Position at the top-left corner
        text=f"Average Sentiment Score: {average_sentiment:.2f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="right"  # Align text to the left
    )

    # Adjust layout with path name in the title
    fig.update_layout(
        xaxis_title="Timestamp (seconds)",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-5, 5]),
        coloraxis_colorbar=dict(title="Sentiment Score")
    )

    # Show the interactive plot
    fig.show()

def calculate_average_sentiment(df):
    return df['analysis.analysis.sentiment_score'].mean()

def main():
    parser = argparse.ArgumentParser(description='Plot sentiment analysis from a JSON file.')
    parser.add_argument('file_path', type=str, help='Path to the JSON data file')
    args = parser.parse_args()

    df = load_data(args.file_path)
    create_plot(df, args.file_path)

    # Calculate and print the average sentiment score
    average_sentiment = calculate_average_sentiment(df)
    print(f"Average Sentiment Score: {average_sentiment:.2f}")

if __name__ == "__main__":
    main()
