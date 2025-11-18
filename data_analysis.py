# Copy and paste this into data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.loaded_successfully = False
        
    def load_dataset(self, dataset_choice='iris'):
        try:
            if dataset_choice == 'iris':
                iris = load_iris()
                self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
                self.df['species'] = [iris.target_names[i] for i in iris.target]
                print("✅ Iris dataset loaded successfully from sklearn!")
            else:
                self.df = pd.read_csv(dataset_choice)
                print(f"✅ Dataset loaded successfully from {dataset_choice}!")
            
            self.loaded_successfully = True
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File {dataset_choice} not found.")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        if not self.loaded_successfully:
            print("No dataset loaded. Please load a dataset first.")
            return
        
        print("=" * 50)
        print("DATASET EXPLORATION")
        print("=" * 50)
        
        print("\n📊 First 5 rows of the dataset:")
        print(self.df.head())
        
        print(f"\n📐 Dataset Shape: {self.df.shape}")
        
        print("\n🔍 Data Types:")
        print(self.df.dtypes)
        
        print("\n🔎 Missing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        if self.df.isnull().sum().sum() > 0:
            print("\n🧹 Cleaning missing values...")
            self.df = self.df.dropna()
            print("Missing values removed.")
        else:
            print("\n✅ No missing values found!")
    
    def basic_analysis(self):
        if not self.loaded_successfully:
            print("No dataset loaded. Please load a dataset first.")
            return
        
        print("\n" + "=" * 50)
        print("BASIC DATA ANALYSIS")
        print("=" * 50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\n📈 Basic Statistics for Numerical Columns:")
        print(self.df[numerical_cols].describe())
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📊 Grouping by '{categorical_cols[0]}' and computing means:")
            grouped_means = self.df.groupby(categorical_cols[0])[numerical_cols].mean()
            print(grouped_means)
            
            print("\n💡 Interesting Findings:")
            max_var_col = grouped_means.std().idxmax()
            print(f"- '{categorical_cols[0]}' shows most variation in '{max_var_col}'")
    
    def create_visualizations(self):
        if not self.loaded_successfully:
            print("No dataset loaded. Please load a dataset first.")
            return
        
        print("\n" + "=" * 50)
        print("DATA VISUALIZATIONS")
        print("=" * 50)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Data Analysis Visualizations', fontsize=16, fontweight='bold')
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Plot 1: Line Chart
        if len(numerical_cols) >= 1:
            axes[0, 0].plot(self.df.index, self.df[numerical_cols[0]], marker='o', linewidth=2, markersize=4, alpha=0.7)
            axes[0, 0].set_title(f'Line Chart: {numerical_cols[0]} Trend', fontweight='bold')
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel(numerical_cols[0])
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Bar Chart
        if len(categorical_cols) > 0 and len(numerical_cols) >= 1:
            grouped_data = self.df.groupby(categorical_cols[0])[numerical_cols[0]].mean()
            bars = axes[0, 1].bar(grouped_data.index, grouped_data.values, 
                                color=sns.color_palette("husl", len(grouped_data)),
                                alpha=0.8, edgecolor='black')
            axes[0, 1].set_title(f'Bar Chart: Average {numerical_cols[0]} by {categorical_cols[0]}', fontweight='bold')
            axes[0, 1].set_xlabel(categorical_cols[0])
            axes[0, 1].set_ylabel(f'Average {numerical_cols[0]}')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 3: Histogram
        if len(numerical_cols) >= 1:
            axes[1, 0].hist(self.df[numerical_cols[0]], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title(f'Histogram: Distribution of {numerical_cols[0]}', fontweight='bold')
            axes[1, 0].set_xlabel(numerical_cols[0])
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter Plot
        if len(numerical_cols) >= 2:
            if len(categorical_cols) > 0:
                scatter = axes[1, 1].scatter(self.df[numerical_cols[0]], self.df[numerical_cols[1]], 
                                            c=pd.factorize(self.df[categorical_cols[0]])[0],
                                            alpha=0.7, s=60, cmap='viridis')
                axes[1, 1].set_title(f'Scatter Plot: {numerical_cols[0]} vs {numerical_cols[1]}', fontweight='bold')
                axes[1, 1].set_xlabel(numerical_cols[0])
                axes[1, 1].set_ylabel(numerical_cols[1])
                
                handles, labels = scatter.legend_elements()
                axes[1, 1].legend(handles, self.df[categorical_cols[0]].unique(), title=categorical_cols[0])
        
        plt.tight_layout()
        plt.savefig('data_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'data_visualizations.png'")

def main():
    analyzer = DataAnalyzer()
    
    print("🎯 PYTHON DATA ANALYSIS ASSIGNMENT")
    print("=" * 40)
    
    success = analyzer.load_dataset('iris')
    
    if success:
        analyzer.explore_data()
        analyzer.basic_analysis()
        analyzer.create_visualizations()
        
        print("\n" + "=" * 50)
        print("✅ ASSIGNMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)

if __name__ == "__main__":
    main()