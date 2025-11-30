import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from src.config import TOXICITY_MODEL, SENTIMENTS, TOXICITY

model_name = TOXICITY_MODEL

toxicity_analyzer = pipeline("text-classification", model=model_name, tokenizer=model_name, device=-1)


def check_toxicity(row):
    text = str(row['Text'])
    existing_sentiment = str(row['sentiment'])


    if existing_sentiment == "Positive":
        return "Safe", 0.0

    # Only run  Negative or Neutral comments
    try:
        # 512 sınır, kelime
        result = toxicity_analyzer(text, truncation=True, max_length=512)[0]
        label = result['label']
        score = result['score']

        if label in ["toxic", "LABEL_1"]:
            return "Toxic", score
        else:
            return "Safe", score

    except Exception:
        return "Error", 0.0

def toxicity_analysis():

    try:
        df = pd.read_csv(SENTIMENTS, lineterminator='\n')
        yield(f"Loaded {len(df)} comments.")
        
        # Check if Column
        if 'sentiment' not in df.columns:
            yield("Error: Your CSV must have a 'sentiment' column!")
            exit()
            
    except FileNotFoundError:
        yield("Error: File not found.")
        exit()

    yield("Analyzing Toxicity (Skipping Positive comments)...")

    # tqdm.pandas()
    results = df.progress_apply(check_toxicity, axis=1)

    df['toxicity_label'] = [res[0] for res in results]
    df['toxicity_score'] = [res[1] for res in results]

    df.to_csv(TOXICITY, index=False)
    yield(f"\nDone! Saved to {TOXICITY}")
