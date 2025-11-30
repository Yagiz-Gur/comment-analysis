def load_stopwords(input_file):

    stopword = set()

    with open (input_file,"r",encoding="utf-8") as f:
        for line in f:
            word = line.replace('"', '').replace(',', '').strip().lower()
            if word:    
                stopword.add(word)

        return stopword
