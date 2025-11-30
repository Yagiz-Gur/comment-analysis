from pathlib import Path

# DIRECTORIES 
FILE_PATH = Path(__file__).resolve()
BASE_DIR = FILE_PATH.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESOURCES_DIR = DATA_DIR / "resources"

#Input And Output Files
COMMENTS = RAW_DIR / "comments.csv"
SENTIMENTS = PROCESSED_DIR / "sentiment.csv"
TOXICITY = PROCESSED_DIR / "toxicity.csv"
STOPWORDS = RESOURCES_DIR / "stopwords.txt"

SENTIMENT_BARCHART =  RESOURCES_DIR / "sentiment_bar.png"
SENTIMENT_HIST = RESOURCES_DIR / "sentiment_hist.png"
SENTIMENT_BOXPLOT = RESOURCES_DIR / "sentiment_box.png"
SENTIMENT_SCATTER = RESOURCES_DIR / "sentiment_scatter.png"

TOXICITY_BARCHART = RESOURCES_DIR / "toxicity_bar.png"
TOXICITY_HIST = RESOURCES_DIR / "toxicity_hist.png"
TOXICITY_BOX = RESOURCES_DIR / "toxicity_box.png"
TOXICITY_SCATTER = RESOURCES_DIR / "toxicity_scatter.png"

# Models
SENTIMENT_MODEL = "kaixkhazaki/turkish-sentiment"
TOXICITY_MODEL = "cagrigungor/turkishtoxic-classifier"
SIMILARITY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

