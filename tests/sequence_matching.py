from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_best_match_segment(pseudo_segment, reference_sentence):
    """
    Finds the best matching text segment in the reference sentence for the given pseudo segment.

    :param pseudo_segment: The segment to match.
    :param reference_sentence: The full reference sentence.
    :param window_size: The number of words in each sliding window segment to compare.
    :return: The best matching segment from the reference sentence.
    """
    window_size = len(pseudo_segment.split())

    # Tokenize the sentences
    pseudo_tokens = pseudo_segment.split()
    reference_tokens = reference_sentence.split()
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer().fit([pseudo_segment, reference_sentence])
    
    # Convert pseudo segment to TF-IDF vector
    pseudo_vector = vectorizer.transform([pseudo_segment])
    
    # Sliding window over reference tokens
    max_similarity = -1
    best_match = ""
    for i in range(len(reference_tokens) - window_size + 1):
        window_segment = " ".join(reference_tokens[i:i+window_size])
        window_vector = vectorizer.transform([window_segment])
        
        # Compute cosine similarity
        similarity = cosine_similarity(pseudo_vector, window_vector)[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = window_segment
    
    return best_match, max_similarity