import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from src.config import SENTIMENT_MODEL, COMMENTS, SENTIMENTS


# Pipeline
sentiment_analyze = pipeline("sentiment-analysis",
    model=SENTIMENT_MODEL,
    tokenizer=SENTIMENT_MODEL)


def get_sentiment(text):
    
    if pd.isna(text) or text == "":
        return None, 0.0
    
    try:
        result = sentiment_analyze(str(text), truncation=True, max_length=512)[0]
        return result['label'], result['score']
    except Exception as e:

        return None, 0.0
    
def sentiment_analysis():

    try:
        df = pd.read_csv(COMMENTS)
        yield("Data loaded ")
    except FileNotFoundError:
        yield("Error: File not found")
        exit()
    
    yield("Analyzing sentiments...")
    # tqdm.pandas()
    df[['sentiment', 'score']] = df["Text"].progress_apply(lambda x: pd.Series(get_sentiment(x)))

    df.to_csv(SENTIMENTS, index=False)

    yield(f"Analysis complete! Saved results to {SENTIMENTS}")
