import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TopicChangeDetector:
    def __init__(self, bert_model_name='bert-base-uncased', n_topics=5):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.model.eval()
        
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

        custom_stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        self.stop_words.update(custom_stops)
        
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_df=0.95, 
            min_df=2,
            stop_words=self.stop_words,
            tokenizer=self._tokenize_and_clean
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
    
    def _tokenize_and_clean(self, text):
        """Improved tokenization and cleaning function"""
        try:
            # 1. Convert to lowercase
            text = text.lower()
            
            # 2. Basic tokenization
            tokens = word_tokenize(text)
            
            # 3. Clean and filter
            cleaned_tokens = []
            for token in tokens:
                # Check if token meets conditions
                if (token not in self.stop_words and  # Not a stopword
                    token.isalpha() and              # Pure alphabetic
                    len(token) > 2 and              # Length > 2
                    not token.isnumeric()):         # Not numeric
                    cleaned_tokens.append(token)
            
            # 4. If too few tokens remain, keep some original tokens
            if len(cleaned_tokens) < 3:
                cleaned_tokens = [t for t in tokens if len(t) > 1 and not t.isnumeric()]
            
            return cleaned_tokens
            
        except Exception as e:
            print(f"Tokenization error: {str(e)}")
            return text.lower().split()
    
    def _extract_topics(self, text_before, text_after):
        """Using LDA to extract topics, increasing the context window and improving text processing"""
        try:
            # 1. Text preprocessing
            text_before = text_before.strip()
            text_after = text_after.strip()
            
            # 2. Expand the context window
            before_tokens = self._tokenize_and_clean(text_before)
            after_tokens = self._tokenize_and_clean(text_after)
            
            # 3. Merge similar words and add synonyms
            before_expanded = self._expand_tokens(before_tokens)
            after_expanded = self._expand_tokens(after_tokens)
            
            # 4. Prepare documents
            docs = [
                ' '.join(before_expanded),
                ' '.join(after_expanded)
            ]
            
            # 5. Adjust CountVectorizer parameters
            self.vectorizer = CountVectorizer(
                max_df=1.0,  # Allow all words
                min_df=1,    # Minimum document frequency is 1
                stop_words=self.stop_words,
                ngram_range=(1, 2),  # Use words and bigrams
                max_features=50      # Limit the number of features
            )
            
            # 6. Generate document-term matrix
            try:
                dtm = self.vectorizer.fit_transform(docs)
                feature_names = self.vectorizer.get_feature_names_out()
                
                if dtm.sum() == 0 or len(feature_names) < 2:
                    raise ValueError("Not enough features for LDA")
                    
            except Exception as e:
                print(f"DTM generation failed: {str(e)}")
                return self._fallback_topic_extraction(before_tokens, after_tokens)
            
            # 7. Configure LDA
            n_topics = min(2, len(feature_names))  # Use fewer topics for short texts
            self.lda = LatentDirichletAllocation(
                n_components=n_topics,
                learning_method='online',
                random_state=42,
                max_iter=20,
                n_jobs=-1
            )
            
            # 8. Fit LDA model
            try:
                lda_output = self.lda.fit_transform(dtm)
                topics_before = self._get_topic_keywords(lda_output[0], feature_names)
                topics_after = self._get_topic_keywords(lda_output[1], feature_names)
                
                return topics_before, topics_after
                
            except Exception as e:
                print(f"LDA fitting failed: {str(e)}")
                return self._fallback_topic_extraction(before_tokens, after_tokens)
            
        except Exception as e:
            print(f"Topic extraction error: {str(e)}")
            return self._fallback_topic_extraction(before_tokens, after_tokens)
    
    def _expand_tokens(self, tokens, min_tokens=5):
        """Expand the list of tokens, adding related words"""
        if len(tokens) < min_tokens:
            # Repeat important words
            importance_sorted = sorted(set(tokens), key=lambda x: len(x), reverse=True)
            expanded = tokens.copy()
            while len(expanded) < min_tokens and importance_sorted:
                expanded.append(importance_sorted[0])
                importance_sorted = importance_sorted[1:]
            return expanded
        return tokens
    
    def _get_topic_keywords(self, topic_distribution, feature_names, n_words=5):
        """Get topic keywords"""
        try:
            # Get the dominant topic
            dominant_topic = topic_distribution.argmax()
            topic_words = self.lda.components_[dominant_topic]
            
            # Get the top N keywords
            top_word_indices = topic_words.argsort()[:-n_words-1:-1]
            keywords = [feature_names[i] for i in top_word_indices]
            
            # Filter keywords
            filtered_keywords = [
                word for word in keywords 
                if len(word) > 2 and word.isalpha()
            ]
            
            return {
                'topic_id': int(dominant_topic),
                'probability': float(topic_distribution[dominant_topic]),
                'keywords': filtered_keywords[:n_words]
            }
        except Exception as e:
            print(f"Error getting topic keywords: {str(e)}")
            return {'topic_id': 0, 'probability': 0.0, 'keywords': []}
    
    def _fallback_topic_extraction(self, before_tokens, after_tokens):
        """Fallback option when LDA fails"""
        return (
            {
                'topic_id': 0,
                'probability': 1.0,
                'keywords': list(set(before_tokens))[:5]
            },
            {
                'topic_id': 1,
                'probability': 1.0,
                'keywords': list(set(after_tokens))[:5]
            }
        )
    
    def _get_bert_embedding(self, text):
        # Get BERT embeddings for English text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]

    def detect_topic_changes(self, text, major_threshold=0.77, minor_threshold=0.83, window_size=3):
        # Split by periods for English sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            print("Not enough sentences to detect topic changes")
            return []
        
        print("\nCalculating BERT embeddings...")
        embeddings = [self._get_bert_embedding(sent + '.') for sent in sentences]
        
        changes = []
        similarities = []
        major_changes = []  # Track positions of major topic changes
        
        # Step 1: Calculate similarities between adjacent sentences
        print("\nCalculating sentence similarities:")
        for i in range(len(sentences) - 1):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
            
            # Detect major topic change, use LDA analysis
            if similarity < major_threshold:
                # Extract topics from before and after text
                topics_before, topics_after = self._extract_topics(
                    sentences[i],
                    sentences[i + 1]
                )
                
                major_changes.append(i)
                changes.append({
                    'position': i + 1,
                    'type': 'major',
                    'similarity': similarity,
                    'sentence_before': sentences[i],
                    'sentence_after': sentences[i + 1],
                    'topic_before': topics_before,
                    'topic_after': topics_after
                })
                print(f"\nComparing sentences {i} and {i+1}:")
                print(f"*** Major topic change detected! ***")
                print(f"Similarity: {similarity:.4f}")
                print(f"Topic before: {topics_before['keywords']}")
                print(f"Topic after: {topics_after['keywords']}")
                continue
        
        # Step 2: Detect minor changes in non-major change areas
        for i in range(len(similarities)):
            if i in major_changes:
                continue
            
            # Get valid context (excluding major changes)
            valid_context = []
            for j in range(max(0, i - window_size), min(len(similarities), i + window_size + 1)):
                if j not in major_changes:
                    valid_context.append(similarities[j])
            
            if not valid_context:
                continue
            
            mean_similarity = np.mean(valid_context)
            std_similarity = np.std(valid_context)
            current_similarity = similarities[i]
            
            # Dynamic threshold for minor changes
            dynamic_threshold = max(
                minor_threshold,
                mean_similarity * 0.95
            )
            
            print(f"\nComparing sentences {i} and {i+1}:")
            print(f"Sentence {i}: {sentences[i]}")
            print(f"Sentence {i+1}: {sentences[i+1]}")
            print(f"Current similarity: {current_similarity:.4f}")
            print(f"Local mean similarity: {mean_similarity:.4f}")
            print(f"Local std: {std_similarity:.4f}")
            print(f"Dynamic threshold: {dynamic_threshold:.4f}")
            
            if (current_similarity < dynamic_threshold and 
                current_similarity >= major_threshold):
                # Extract topics from before and after text
                topics_before, topics_after = self._extract_topics(
                    sentences[i],
                    sentences[i + 1]
                )
                
                changes.append({
                    'position': i + 1,
                    'type': 'minor',
                    'similarity': current_similarity,
                    'local_mean': mean_similarity,
                    'dynamic_threshold': dynamic_threshold,
                    'sentence_before': sentences[i],
                    'sentence_after': sentences[i + 1],
                    'topic_before': topics_before,
                    'topic_after': topics_after
                })
        
        print(f"\nTotal topic changes detected: {len(changes)}")
        print(f"Major changes: {len([c for c in changes if c['type'] == 'major'])}")
        print(f"Minor changes: {len([c for c in changes if c['type'] == 'minor'])}")
        return changes

def test_lda_detection():
    """Test function for LDA topic detection"""
    # Create detector instance
    detector = TopicChangeDetector()
    
    # Prepare test texts
    test_texts = [
        # AI and machine learning related text
        """Deep learning has revolutionized artificial intelligence in recent years.
        Neural networks are becoming increasingly sophisticated and powerful.
        These models can now handle complex tasks with remarkable accuracy.""",
        
        # Zoo related text
        """The pandas at the zoo were absolutely adorable today.
        They were playing with bamboo and rolling around happily.
        The zookeeper gave an interesting presentation about their diet.""",
        
        # Back to AI topic
        """Transformer models have changed natural language processing.
        BERT and GPT have shown remarkable performance in various tasks.
        These language models can understand context very well."""
    ]
    
    # Test topic extraction
    print("\n=== Testing LDA Topic Detection ===\n")
    
    for i in range(len(test_texts) - 1):
        print(f"\nComparing Text {i+1} and Text {i+2}:")
        print("-" * 50)
        
        # Extract topics
        topics_before, topics_after = detector._extract_topics(
            test_texts[i],
            test_texts[i + 1]
        )
        
        # Print results
        print("\nText 1:")
        print(test_texts[i][:100] + "...")
        print("Topics:", ', '.join(topics_before['keywords']))
        print("\nText 2:")
        print(test_texts[i + 1][:100] + "...")
        print("Topics:", ', '.join(topics_after['keywords']))
        
        # Calculate similarity
        similarity = cosine_similarity(
            detector._get_bert_embedding(test_texts[i]).reshape(1, -1),
            detector._get_bert_embedding(test_texts[i + 1]).reshape(1, -1)
        )[0][0]
        
        print(f"\nSimilarity Score: {similarity:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    # Run test
    test_lda_detection()
    