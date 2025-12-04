import streamlit as st
import pandas as pd 
from src.youtube_client import save_comments_to_csv
from src.sentiment_analysis import sentiment_analysis
from src.toxicity import toxicity_analysis
from src.visulize import visulize_words, visulize_sentiment, visulize_toxicity
from src.config import SENTIMENT_SCATTER, SENTIMENT_BARCHART, SENTIMENT_BOXPLOT, SENTIMENT_HIST
from src.config import TOXICITY_BARCHART, TOXICITY_BOX, TOXICITY_HIST, TOXICITY_SCATTER
from src.config  import COMMENTS
import os 

def init_page():
    """Sayfa ayarlarını ve CSS stillerini yükler."""
    st.set_page_config(
        page_title="YouTube Analiz Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS 
    st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎥 YouTube Yorum Analiz Paneli")
    st.markdown("---")

#SIDEBAr
def render_sidebar():
    """Yan paneli oluşturur ve girilen URL'yi döndürür."""
    with st.sidebar:
        st.header("⚙️ Ayarlar & Giriş")
        url = st.text_input("YouTube Video Linki", placeholder="https://youtube.com/...")
        
        if 'data_downloaded' not in st.session_state:
            st.session_state.data_downloaded = False
        if 'analysis_done' not in st.session_state:
            st.session_state.analysis_done = False

        st.info("💡 Linki girdikten sonra 'Verileri Getir' butonuna basın.")
        
        return url

def tab_download(url):
    """İndirme sekmesinin içeriği."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if url:
            st.video(url)
    
    with col2:
        st.subheader("İşlem Merkezi")
        if st.button("🚀 Yorumları İndir"):
            if not url:
                st.warning("Lütfen geçerli bir link giriniz.")
            else:
                with st.status("Yorumlar indiriliyor...", expanded=True) as status:
                    st.write("Bağlantı kuruluyor...")
                    for message in save_comments_to_csv(url):
                        st.write(f"Log: {message}")
                    
                    status.update(label="İndirme Tamamlandı!", state="complete", expanded=False)
                
                st.session_state.data_downloaded = True
                st.success("✅ Tüm yorumlar başarıyla kaydedildi.")

        if st.session_state.data_downloaded:
            st.info("Veri seti analize hazır. Lütfen 'Yapay Zeka Analizi' sekmesine geçin.")

def tab_analysis():
    """Analiz sekmesinin içeriği."""
    st.subheader("Model Analizleri")
    
    if not st.session_state.data_downloaded:
        st.warning("⚠️ Lütfen önce 1. Sekmeden yorumları indirin.")
        return

    st.markdown("## Duygu ve Toxicity analizi")
    st.write("Sentiment Analysis, bir metnin olumlu, olumsuz veya nötr duygu taşıyıp taşımadığını belirleyen bir" \
    " yapay zekâ analizidir. Toxicity Analysis ise metindeki hakaret, küfür, tehdit veya aşağılayıcı ifadeleri" \
    " tespit ederek zararlı içeriği ortaya çıkarır. Bu iki analiz birlikte, kullanıcıların daha güvenli ve" \
    " anlamlı bir iletişim deneyimi yaşamasını sağlar.")

    if st.button("Analizi Başlat"):
        try:
            for message in sentiment_analysis():
                st.write(message)
            st.info("Sentiment analysis tamamlandı")
        except FileNotFoundError:
            st.error("Dosya bulunamadı !")
        try:
            for message in toxicity_analysis():
                st.write(message)
            st.info("Toxicity analysis tamamlandı")
        except:
            st.info("Hata oluştu")
        
        show_metrics()

def show_metrics():
    # 1. Load the Data
    try:
        # Make sure this matches your actual file name
        df = pd.read_csv("data/processed/toxicity.csv") 
    except FileNotFoundError:
        st.error("Error: 'Toxicity.csv' not found. Please run the analysis first.")
        return

    # 2. Basic Calculations
    total_comments = len(df)
    
    # Calculate Sentiment Counts (Safe method using .get to avoid errors if a label is missing)
    sentiment_counts = df['sentiment'].value_counts()
    negative_count = sentiment_counts.get('Negative', 0)
    positive_count = sentiment_counts.get('Positive', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    # Calculate Sentiment Percentage
    neg_percentage = (negative_count / total_comments) * 100 if total_comments > 0 else 0

    # Calculate Toxicity Counts
    # Assuming the column is named 'toxicity_label' and the bad label is 'Toxic'
    toxicity_counts = df['toxicity_label'].value_counts()
    toxic_count = toxicity_counts.get('Toxic', 0) # Change 'Toxic' to whatever your label is (e.g., 'severe_toxicity')
    
    # Calculate Toxicity Percentage
    toxic_percentage = (toxic_count / total_comments) * 100 if total_comments > 0 else 0

    # 3. Display in Streamlit
    st.header("📊 Analysis Overview")
    
    # Row 1: Big Summary Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Comments", value=total_comments)
    
    with col2:
        st.metric(
            label="Negative Comments", 
            value=f"{negative_count}", 
            delta=f"{neg_percentage:.1f}% of total",
            delta_color="inverse" # Makes red color for negative things
        )
        
    with col3:
        st.metric(
            label="Toxic Comments", 
            value=f"{toxic_count}", 
            delta=f"{toxic_percentage:.1f}% of total",
            delta_color="inverse"
        )

    # Row 2: Detailed Breakdown (Optional but nice)
    st.subheader("Detailed Breakdown")
    col4, col5 = st.columns(2)
    
    with col4:
        st.write("**Sentiment Distribution**")
        st.dataframe(sentiment_counts, width="stretch")
        # Or use a chart: st.bar_chart(sentiment_counts)
        
    with col5:
        st.write("**Toxicity Distribution**")
        st.dataframe(toxicity_counts, width="stretch")
        # Or use a chart: st.bar_chart(toxicity_counts)   

def tab_visualization():    

    if st.button("📊 Grafikleri Oluştur / Yenile"):
        with st.spinner('Grafikler çiziliyor...'):
            try:
                visulize_sentiment()
                visulize_toxicity()
                
                st.markdown("### 🧬 Toksisite Raporu")
                
                # Üst Sıra
                row1_col1, row1_col2 = st.columns(2)
                with row1_col1:
                    st.image(TOXICITY_BARCHART, width="stretch", caption="Toksik/Temiz Dağılımı")
                    with st.expander("Detaylı Açıklama"):
                        st.write("Veri setindeki toplam toksik ve temiz yorumların karşılaştırması.")

                with row1_col2:
                    st.image(TOXICITY_BOX, width="stretch", caption="Güven Skoru Analizi")
                    with st.expander("Detaylı Açıklama"):
                        st.write("Modelin verdiği kararlardaki güven aralığı dağılımı.")

                st.divider()

                # Alt Sıra
                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.image(TOXICITY_HIST, width="stretch", caption="Yoğunluk Haritası")
                
                with row2_col2:
                    st.image(TOXICITY_SCATTER, width="stretch", caption="Etkileşim Analizi")

                # Sentiment Grafikle
                st.markdown("### 😊 Duygu Durum Raporu")
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.image(SENTIMENT_BARCHART, width="stretch")
                with s_col2:
                    st.image(SENTIMENT_SCATTER, width="stretch")
            except FileNotFoundError:
                st.error("Dosyo bulunamadı!")

def comments():
    st.header("Ham Veri İnceleme")
    st.markdown("---")

    if not os.path.exists(COMMENTS):
        st.warning(f"⚠️ Veri dosyası bulunamadı: `{COMMENTS}`")
        st.info("Lütfen önce ana sayfadan yorumları indiriniz.")
        return

    try:

        df = pd.read_csv(COMMENTS)

        col1, col2 = st.columns(2)
        col1.metric("Toplam Yorum", df.shape[0])


        search = col2.text_input("🔍 Tabloda Ara", placeholder="Kelime yazın...")
        
        if search:
            mask = df.apply(lambda x: x.astype(str).str.contains(search, case=False).any(), axis=1)
            df_display = df[mask]
        else:
            df_display = df

        st.markdown("### Yorum Listesi")
        st.dataframe(
            df_display, 
            width="stretch", 
            height=600,          
            hide_index=True          
        )

        # CSV İndrme Buton
        st.download_button(
            label="💾 Bu Tabloyu İndir (CSV)",
            data=df_display.to_csv(index=False).encode('utf-8'),
            file_name='filtrelenmis_yorumlar.csv',
            mime='text/csv',
        )
 
        st.markdown("---")
        fig1, fig2 = visulize_words()
        st.image(fig1)
        st.image(fig2)

    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}")


def main():
    init_page()
    url = render_sidebar()
    
    t1, t2, t3, t4 = st.tabs(["📥 Veri İndirme","Yorumlar", "🧠 Yapay Zeka Analizi", "📊 Görsel Rapor"])
    
    with t1:
        tab_download(url)
    
    with t2:
        comments()
    with t3:
        tab_analysis()
        
    with t4:
        tab_visualization()

if __name__ == "__main__":
    main()