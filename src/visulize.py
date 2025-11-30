import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from src.preprocessing import load_stopwords
from src.config import COMMENTS, STOPWORDS, TOXICITY
from src.config import SENTIMENT_BARCHART, SENTIMENT_BOXPLOT,SENTIMENT_HIST,SENTIMENT_SCATTER
from src.config import TOXICITY_BARCHART, TOXICITY_HIST,TOXICITY_BOX,TOXICITY_SCATTER


def visulize_words():

    stopwordss = {w.strip().lower() for w in load_stopwords(STOPWORDS)}

    df = pd.read_csv(COMMENTS ,lineterminator='\n')
    text_data = df['Text'].dropna().astype(str).tolist()

    all_words = []

    for comment in text_data:
        # Lowercase
        comment = comment.lower()
        #regex 
        comment = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜı\s]', ' ', comment) 

        # Split into words
        words = comment.split()
        
        # Filter  stopwords and short words
        clean_words = [w for w in words if w.lower() not in stopwordss and len(w) > 2]

        all_words.extend(clean_words)

    word_counts = Counter(all_words)
    common_words = word_counts.most_common(20)

    words = [w[0] for w in common_words]
    counts = [w[1] for w in common_words]

    barchart_path="data/processed/word_frequency_bar.png"

    plt.figure(figsize=(12, 6))
    plt.barh(words, counts, color='skyblue')
    plt.xlabel("Frequency (Count)")
    plt.ylabel("Words")
    plt.title("Top 20 Most Common Words in Comments")
    plt.gca().invert_yaxis()  
    plt.savefig(barchart_path)
    plt.close()


    text_for_cloud = " ".join(all_words)
    wordcloud_path = "data/processed/word_cloud.png"

    # Create cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        stopwords=stopwordss,
        background_color='white', 
        colormap='viridis' 
    ).generate(text_for_cloud)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off") # Turn off axisnum
    plt.title("Word Cloud of Comments")
    plt.savefig(wordcloud_path)
    plt.close()

    return wordcloud_path, barchart_path



def visulize_sentiment():
    df = pd.read_csv(TOXICITY)

    sns.set(style="whitegrid")

    #Barchar
    plt.figure(figsize=(10, 6)) 
    sns.countplot(x='sentiment', data=df, palette='viridis')
    plt.title('Duygu Sayı Dağılımı')
    plt.ylabel('Adet')

    plt.savefig(SENTIMENT_BARCHART, dpi=300, bbox_inches='tight')
    plt.close()

    # HİSTOGRAM
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='score', kde=True, color='skyblue', bins=20)
    plt.title('Sentiment Score (Güven) Dağılımı')
    
    plt.savefig(SENTIMENT_HIST, dpi=300, bbox_inches='tight')
    plt.close()

    #box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment', y='score', data=df, palette='viridis')
    plt.title('Duygu Kategorilerine Göre Skorlar')
    
    plt.savefig(SENTIMENT_BOXPLOT, dpi=300, bbox_inches='tight')
    plt.close()

    # scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='score', y='Likes', hue='sentiment', data=df, alpha=0.7)
    plt.title('Score ve Like İlişkisi')

    plt.savefig(SENTIMENT_SCATTER, dpi=300, bbox_inches='tight')
    plt.close()


def visulize_toxicity():

    df = pd.read_csv(TOXICITY)

    sns.set(style="whitegrid")

    #bar charth
    plt.figure(figsize=(10, 6))
    sns.countplot(x='toxicity_label', data=df, palette='Reds_d')
    plt.title('Toksisite Etiket Dağılımı')
    plt.ylabel('Yorum Sayısı')
    plt.xlabel('Etiket')
 
    plt.savefig(TOXICITY_BARCHART, dpi=300, bbox_inches='tight')
    plt.close()

    #histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='toxicity_score', kde=True, color='red', bins=30)
    plt.title('Toksisite Skoru Dağılımı (0=Temiz, 1=Toksik)')
    plt.xlabel('Toksisite Puanı')    

    plt.savefig(TOXICITY_HIST, dpi=300, bbox_inches='tight')
    plt.close()

    #Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='toxicity_label', y='toxicity_score', data=df, palette='Reds')
    plt.title('Etiket Bazında Skor Güveni')
    

    plt.savefig(TOXICITY_BOX, dpi=300, bbox_inches='tight')
    plt.close()

    #sctter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='toxicity_score', y='Likes', hue='toxicity_label', data=df, palette='Reds', alpha=0.7)
    plt.title('Toksisite Skoru ve Like İlişkisi')
    

    plt.savefig(TOXICITY_SCATTER, dpi=300, bbox_inches='tight')
    plt.close()
