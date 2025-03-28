import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
import string
import numpy as np
from scipy.stats import norm

# Load CMU Pronouncing Dictionary
pronunciation_dict = cmudict.dict()

def syllable_count(word):
    """Determine the number of syllables in a word."""
    word = word.lower()
    if word in pronunciation_dict:
        return max([len([phoneme for phoneme in phonetic if phoneme[-1].isdigit()]) for phonetic in pronunciation_dict[word]])
    return 1  # Assume one syllable if the word isn't found

def is_polysyllabic(word):
    """Identify if a word is polysyllabic (i.e., has 3 or more syllables)."""
    return syllable_count(word) >= 3

def calculate_complexity_index(text):
    """
    Calculate a complexity index (0-100) based on Flesch-Kincaid grade level and percentage of complex words.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        float: Complexity index from 0-100
    """
    # Handle empty text
    if not text or not text.strip():
        return 0
    
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    
    sentence_count = max(1, len(sentences))
    word_count = max(1, len(tokens))
    
    # Calculate Flesch-Kincaid Grade Level using the same formula as the reference code
    total_syllables = sum(syllable_count(token) for token in tokens)
    fk_grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59
    
    # Calculate percentage of complex words
    complex_word_count = sum(1 for token in tokens if is_polysyllabic(token))
    percent_complex_words = (complex_word_count / word_count) * 100
    
    # Cap FK grade at 14 (college level)
    fk_grade_level = min(fk_grade_level, 14)
    
    # Cap percent complex at 20%
    percent_complex_words = min(percent_complex_words, 20)
    
    # Normalize scores to 0-100 range
    fk_normalized = (fk_grade_level / 14) * 100
    complex_normalized = (percent_complex_words / 20) * 100
    
    # Average the two normalized scores for the final complexity index
    complexity_index = (fk_normalized + complex_normalized) / 2
    
    return round(complexity_index, 2)

# Calculates a slop score for a provided text

import json
import re
import numpy as np
from joblib import Parallel, delayed

def load_and_preprocess_slop_words():
    with open('data/slop_phrase_prob_adjustments.json', 'r') as f:
        slop_phrases = json.load(f)
    
    phrase_weighting = [1.0 - prob_adjustment for word, prob_adjustment in slop_phrases]
    max_score = max(phrase_weighting)
    scaled_weightings = [score / max_score for score in phrase_weighting]
    n_slop_words = 600
    return {word.lower(): score for (word, _), score in zip(slop_phrases[:n_slop_words], scaled_weightings[:n_slop_words])}

def extract_text_blocks(file_path, compiled_pattern):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    matches = compiled_pattern.findall(content)
    return '\n'.join(matches)

def calculate_slop_score_chunk(args):
    text, slop_words_chunk = args
    return sum(
        score * len(re.findall(r'\b' + re.escape(word) + r'\b', text))
        for word, score in slop_words_chunk.items()
    )

def split_into_chunks(slop_words, num_chunks):
    slop_words_items = list(slop_words.items())
    chunk_size = len(slop_words_items) // num_chunks
    if chunk_size == 0:
        chunk_size = 1
    return [dict(slop_words_items[i:i + chunk_size]) for i in range(0, len(slop_words_items), chunk_size)]


# Call this to function to calculate a slop score.
# This is the way it's calculated for the eqbench creative writing leaderboard.
def calculate_slop_index(extracted_text):    
    slop_words = load_and_preprocess_slop_words()
    
    num_chunks = 12 #mp.cpu_count()
    slop_words_chunks = split_into_chunks(slop_words, num_chunks)
    
    if not extracted_text:
        slop_index = 0.0
    else:
        # Parallelize the calculation using joblib
        slop_scores = Parallel(n_jobs=num_chunks)(delayed(calculate_slop_score_chunk)((extracted_text, chunk)) for chunk in slop_words_chunks)
        
        slop_score = sum(slop_scores)
        total_words = len(extracted_text.split())
        slop_index = (slop_score / total_words) * 1000 if total_words > 0 else 0
    return slop_index



