# cord19_analysis.py - Data Analysis Module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

class CORD19Analyzer:
    def __init__(self):
        self.df = None
        self.loaded_successfully = False
    
    def load_data(self, filepath=None):
        try:
            if filepath:
                self.df = pd.read_csv(filepath)
                print(f"✅ CORD-19 dataset loaded from {filepath}")
            else:
                self.df = self._create_sample_data()
                print("✅ Sample CORD-19 data loaded")
            
            self._clean_data()
            self.loaded_successfully = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _create_sample_data(self):
        sample_data = {
            'title': [
                'COVID-19 transmission dynamics and control measures',
                'Clinical characteristics of coronavirus patients',
                'Vaccine development for SARS-CoV-2',
                'Social distancing effectiveness in pandemic control',
                'Economic impact of COVID-19 lockdown measures',
                'Mental health during pandemic isolation',
                'Treatment protocols for severe COVID-19 cases',
                'Viral mutation patterns and vaccine efficacy'
            ],
            'abstract': [
                'Study of transmission patterns and effectiveness of various control measures.',
                'Analysis of clinical features in confirmed coronavirus cases.',
                'Review of current vaccine development approaches and challenges.',
                'Evaluation of social distancing impact on infection rates.',
                'Assessment of economic consequences from lockdown policies.',
                'Examination of mental health effects during prolonged isolation.',
                'Development of treatment guidelines for severe cases.',
                'Investigation of viral mutations and their impact on vaccines.'
            ],
            'publish_time': [
                '2020-03-15', '2020-02-28', '2020-04-10', '2020-03-22',
                '2020-05-01', '2020-04-18', '2020-03-30', '2020-04-25'
            ],
            'authors': [
                'Smith A, Johnson B', 'Lee C, Wang D', 'Garcia E', 'Brown F, Davis G',
                'Wilson H, Taylor I', 'Martinez J', 'Anderson K, Thomas L', 'Clark M'
            ],
            'journal': [
                'Journal of Epidemiology', 'Clinical Medicine', 'Vaccine Research',
                'Public Health', 'Economics Review', 'Psychology Today',
                'Medical Protocols', 'Virology Journal'
            ],
            'has_pdf_parse': [True, True, False, True, False, True, True, False],
            'has_pmc_xml_parse': [True, False, True, True, False, True, False, True]
        }
        return pd.DataFrame(sample_data)
    
    def _clean_data(self):
        self.df['publish_time'] = pd.to_datetime(self.df['publish_time'], errors='coerce')
        self.df['abstract'] = self.df['abstract'].fillna('No abstract available')
        self.df['authors'] = self.df['authors'].fillna('Unknown authors')
        
        self.df['abstract_length'] = self.df['abstract'].str.len()
        self.df['month'] = self.df['publish_time'].dt.to_period('M')
    
    def basic_statistics(self):
        if not self.loaded_successfully:
            print("No data loaded")
            return
        
        stats = {
            'total_papers': len(self.df),
            'date_range': f"{self.df['publish_time'].min().strftime('%Y-%m-%d')} to {self.df['publish_time'].max().strftime('%Y-%m-%d')}",
            'unique_journals': self.df['journal'].nunique(),
            'papers_with_pdf': self.df['has_pdf_parse'].sum(),
            'papers_with_pmc': self.df['has_pmc_xml_parse'].sum(),
            'avg_abstract_length': self.df['abstract_length'].mean()
        }
        
        return stats
    
    def create_visualizations(self):
        if not self.loaded_successfully:
            print("No data loaded")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CORD-19 Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Publication timeline
        monthly_counts = self.df.groupby('month').size()
        axes[0, 0].bar(range(len(monthly_counts)), monthly_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Publications per Month')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Number of Papers')
        axes[0, 0].set_xticks(range(len(monthly_counts)))
        axes[0, 0].set_xticklabels([str(period) for period in monthly_counts.index], rotation=45)
        
        # Plot 2: Journal distribution
        journal_counts = self.df['journal'].value_counts().head(5)
        axes[0, 1].pie(journal_counts.values, labels=journal_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Top 5 Journals')
        
        # Plot 3: Abstract length distribution
        axes[1, 0].hist(self.df['abstract_length'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Abstract Length Distribution')
        axes[1, 0].set_xlabel('Abstract Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Data availability
        data_availability = {
            'PDF Available': self.df['has_pdf_parse'].sum(),
            'PMC XML Available': self.df['has_pmc_xml_parse'].sum()
        }
        axes[1, 1].bar(data_availability.keys(), data_availability.values(), color=['orange', 'purple'])
        axes[1, 1].set_title('Data Availability')
        axes[1, 1].set_ylabel('Number of Papers')
        
        plt.tight_layout()
        plt.savefig('cord19_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'cord19_analysis.png'")

def main():
    analyzer = CORD19Analyzer()
    
    print("🔬 CORD-19 Research Dataset Analysis")
    print("=" * 40)
    
    success = analyzer.load_data()
    
    if success:
        stats = analyzer.basic_statistics()
        print("\n📊 BASIC STATISTICS:")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n🎨 GENERATING VISUALIZATIONS...")
        analyzer.create_visualizations()
        
        print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()