# nlp_spacy.py - NLP Analysis with spaCy
import spacy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

class AmazonReviewAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Please install: python -m spacy download en_core_web_sm")
            raise
        
        self.positive_words = {'excellent', 'amazing', 'great', 'good', 'awesome', 'love', 'perfect'}
        self.negative_words = {'terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing'}
    
    def generate_sample_reviews(self):
        sample_reviews = [
            "I bought the Apple iPhone and it's absolutely amazing! The camera quality is excellent.",
            "The Samsung phone I received was defective. The screen had dead pixels. Very disappointing.",
            "Microsoft Surface Pro is a fantastic device for work. The performance is brilliant.",
            "This Sony headphones are terrible. The sound quality is poor and they broke quickly.",
            "Google Pixel has an awesome camera and the Android experience is perfect."
        ]
        return sample_reviews
    
    def perform_ner_analysis(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities, doc
    
    def analyze_sentiment(self, text):
        doc = self.nlp(text.lower())
        positive_count = 0
        negative_count = 0
        
        for token in doc:
            if token.text in self.positive_words:
                positive_count += 1
            elif token.text in self.negative_words:
                negative_count += 1
        
        if positive_count > negative_count:
            sentiment = "POSITIVE"
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = "NEGATIVE" 
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def run_complete_analysis(self):
        print("🔍 Performing NLP Analysis on Amazon Reviews...")
        reviews = self.generate_sample_reviews()
        results = []
        
        for i, review in enumerate(reviews, 1):
            print(f"\n📝 Review {i}: {review}")
            entities, doc = self.perform_ner_analysis(review)
            sentiment_result = self.analyze_sentiment(review)
            
            result = {
                'review_id': i,
                'review_text': review,
                'entities': entities,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence']
            }
            results.append(result)
            
            print(f"  🏷️  Entities found: {len(entities)}")
            for entity in entities:
                print(f"     - {entity['text']} ({entity['label']})")
            print(f"  😊 Sentiment: {sentiment_result['sentiment']} (Confidence: {sentiment_result['confidence']:.2f})")
        
        self.generate_summary_statistics(results)
        return results
    
    def generate_summary_statistics(self, results):
        print("\n" + "="*50)
        print("📈 NLP ANALYSIS SUMMARY")
        print("="*50)
        
        sentiment_counts = Counter([r['sentiment'] for r in results])
        total_reviews = len(results)
        
        print(f"Total Reviews Analyzed: {total_reviews}")
        print(f"Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_reviews) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        # Create visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sentiment distribution
        ax[0].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), 
                 autopct='%1.1f%%', startangle=90)
        ax[0].set_title('Sentiment Distribution')
        
        # Entity types
        all_entities = []
        for result in results:
            all_entities.extend([ent['label'] for ent in result['entities']])
        
        entity_counts = Counter(all_entities)
        if entity_counts:
            ax[1].bar(entity_counts.keys(), entity_counts.values())
            ax[1].set_title('Named Entity Types')
            ax[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('nlp_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    nlp_analyzer = AmazonReviewAnalyzer()
    results = nlp_analyzer.run_complete_analysis()
    
    print("\n" + "="*50)
    print("🎉 NLP PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)