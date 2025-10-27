"""Data quality and augmentation using cleanlab, nlpaug, textstat"""

import numpy as np
import pandas as pd
from typing import List, Dict
import cleanlab
from cleanlab.filter import find_label_issues
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import textstat
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression


class DataQualityChecker:
    """Data quality analysis using cleanlab"""
    
    def __init__(self):
        self.issues = []
        
    def find_label_errors(self, texts: List[str], labels: List[int]):
        """Detect mislabeled data using cleanlab"""
        
        # Create simple features (placeholder)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        
        # Get predicted probabilities
        model = LogisticRegression(max_iter=1000)
        pred_probs = cross_val_predict(
            model, X, labels,
            cv=5,
            method='predict_proba'
        )
        
        # Find label issues
        ranked_label_issues = find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )
        
        self.issues = ranked_label_issues
        return ranked_label_issues
    
    def get_quality_scores(self, texts: List[str]) -> pd.DataFrame:
        """Compute quality metrics for texts"""
        
        scores = []
        for text in texts:
            score = {
                'length': len(text),
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'text_standard': textstat.text_standard(text, float_output=True),
                'syllable_count': textstat.syllable_count(text),
                'lexicon_count': textstat.lexicon_count(text),
                'sentence_count': textstat.sentence_count(text),
            }
            scores.append(score)
        
        return pd.DataFrame(scores)
    
    def filter_low_quality(self, texts: List[str], threshold: float = 30.0) -> List[str]:
        """Filter out low-quality texts"""
        
        quality_df = self.get_quality_scores(texts)
        
        # Keep texts with reasonable reading ease
        mask = (quality_df['flesch_reading_ease'] > threshold) & \
               (quality_df['length'] > 50)
        
        filtered = [text for i, text in enumerate(texts) if mask[i]]
        
        print(f"Filtered {len(texts) - len(filtered)} low-quality samples")
        return filtered


class TextAugmenter:
    """Text augmentation using nlpaug"""
    
    def __init__(self):
        # Synonym augmenter
        self.syn_aug = naw.SynonymAug(aug_src='wordnet')
        
        # Contextual word embeddings augmenter
        self.context_aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute"
        )
        
        # Back translation augmenter
        self.back_trans_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en'
        )
        
    def augment_synonym(self, text: str, n: int = 1) -> List[str]:
        """Augment using synonyms"""
        return self.syn_aug.augment(text, n=n)
    
    def augment_contextual(self, text: str, n: int = 1) -> List[str]:
        """Augment using contextual embeddings"""
        return self.context_aug.augment(text, n=n)
    
    def augment_back_translation(self, text: str) -> str:
        """Augment using back translation"""
        return self.back_trans_aug.augment(text)
    
    def augment_dataset(self, texts: List[str], method='synonym', n=1) -> List[str]:
        """Augment entire dataset"""
        
        augmented = []
        for text in texts:
            if method == 'synonym':
                aug_texts = self.augment_synonym(text, n=n)
            elif method == 'contextual':
                aug_texts = self.augment_contextual(text, n=n)
            elif method == 'back_translation':
                aug_texts = [self.augment_back_translation(text)]
            else:
                aug_texts = [text]
            
            augmented.extend(aug_texts if isinstance(aug_texts, list) else [aug_texts])
        
        return augmented


class DatasetCleaner:
    """Complete dataset cleaning pipeline"""
    
    def __init__(self):
        self.quality_checker = DataQualityChecker()
        self.augmenter = TextAugmenter()
        
    def clean_and_augment(
        self,
        texts: List[str],
        labels: List[int] = None,
        augment: bool = True,
        quality_threshold: float = 30.0
    ) -> Dict:
        """Complete cleaning and augmentation pipeline"""
        
        print(f"Original dataset size: {len(texts)}")
        
        # 1. Quality filtering
        filtered_texts = self.quality_checker.filter_low_quality(
            texts, threshold=quality_threshold
        )
        
        # 2. Find label errors if labels provided
        if labels is not None:
            label_issues = self.quality_checker.find_label_errors(
                filtered_texts, labels
            )
            print(f"Found {len(label_issues)} potential label issues")
        
        # 3. Augmentation
        if augment:
            augmented_texts = self.augmenter.augment_dataset(
                filtered_texts[:100],  # Augment subset for speed
                method='synonym',
                n=2
            )
            print(f"Augmented dataset size: {len(augmented_texts)}")
        else:
            augmented_texts = filtered_texts
        
        # 4. Quality scores
        quality_scores = self.quality_checker.get_quality_scores(
            augmented_texts[:100]
        )
        
        return {
            'texts': augmented_texts,
            'quality_scores': quality_scores,
            'label_issues': label_issues if labels else None,
            'stats': {
                'original_size': len(texts),
                'filtered_size': len(filtered_texts),
                'final_size': len(augmented_texts)
            }
        }
