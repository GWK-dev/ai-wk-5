# classical_ml_sklearn.py - Iris Classification with Decision Tree
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class IrisClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.iris = load_iris()
        
    def load_and_preprocess_data(self):
        print("📊 Loading Iris Dataset...")
        X = self.iris.data
        y = self.iris.target
        
        print(f"Dataset shape: {X.shape}")
        print(f"Missing values: {np.isnan(X).sum()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        print("🤖 Training Decision Tree Classifier...")
        self.model.fit(X_train, y_train)
        print("✅ Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        print("📈 Evaluating Model Performance...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📈 Recall: {recall:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.iris.target_names))
        
        return accuracy, precision, recall
    
    def visualize_tree(self):
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, 
                 feature_names=self.iris.feature_names,
                 class_names=self.iris.target_names,
                 filled=True,
                 rounded=True)
        plt.title("Decision Tree - Iris Species Classification")
        plt.tight_layout()
        plt.savefig('decision_tree_iris.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_complete_analysis(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        self.train_model(X_train, y_train)
        accuracy, precision, recall = self.evaluate_model(X_test, y_test)
        self.visualize_tree()
        
        return accuracy, precision, recall

if __name__ == "__main__":
    iris_classifier = IrisClassifier()
    accuracy, precision, recall = iris_classifier.run_complete_analysis()
    
    print("\n" + "="*50)
    print("🎉 CLASSICAL ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")