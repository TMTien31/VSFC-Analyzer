def split_sentences(text):
    raw_sentences = text.split('.')
    clean_sentences = [s.strip() for s in raw_sentences if s.strip()]
    return clean_sentences
