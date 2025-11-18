# app.py - Streamlit Application for CORD-19 Analysis
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Research Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_data():
    """Load sample data for demonstration"""
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

def main():
    st.title("🔬 CORD-19 Research Paper Analysis")
    st.markdown("### COVID-19 Open Research Dataset Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Analysis Section", 
                                   ["Data Overview", "Publication Trends", "Content Analysis", "Research Insights"])
    
    # Load data
    st.sidebar.info("📊 Loading sample CORD-19 metadata...")
    df = load_sample_data()
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    
    if app_mode == "Data Overview":
        show_data_overview(df)
    elif app_mode == "Publication Trends":
        show_publication_trends(df)
    elif app_mode == "Content Analysis":
        show_content_analysis(df)
    elif app_mode == "Research Insights":
        show_research_insights(df)

def show_data_overview(df):
    st.header("📋 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Statistics")
        st.write(f"**Total Papers:** {len(df)}")
        st.write(f"**Date Range:** {df['publish_time'].min().strftime('%Y-%m-%d')} to {df['publish_time'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Journals Represented:** {df['journal'].nunique()}")
        st.write(f"**Papers with PDF:** {df['has_pdf_parse'].sum()}")
        st.write(f"**Papers with PMC XML:** {df['has_pmc_xml_parse'].sum()}")
    
    with col2:
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Sample Paper Details")
    selected_paper = st.selectbox("Select a paper to view details:", df['title'])
    paper_details = df[df['title'] == selected_paper].iloc[0]
    
    st.write(f"**Title:** {paper_details['title']}")
    st.write(f"**Authors:** {paper_details['authors']}")
    st.write(f"**Journal:** {paper_details['journal']}")
    st.write(f"**Publication Date:** {paper_details['publish_time'].strftime('%Y-%m-%d')}")
    st.write(f"**Abstract:** {paper_details['abstract']}")

def show_publication_trends(df):
    st.header("📈 Publication Trends")
    
    # Monthly publication count
    st.subheader("Publication Timeline")
    monthly_counts = df.groupby(df['publish_time'].dt.to_period('M')).size()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Publications per Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Papers')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Journal distribution
    st.subheader("Journal Distribution")
    journal_counts = df['journal'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        journal_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Papers by Journal')
        ax.set_ylabel('')
        st.pyplot(fig)
    
    with col2:
        st.dataframe(journal_counts, use_container_width=True)

def show_content_analysis(df):
    st.header("📊 Content Analysis")
    
    # Word frequency in titles
    st.subheader("Research Focus Areas")
    
    keywords = ['transmission', 'clinical', 'vaccine', 'social', 'economic', 'mental', 'treatment', 'mutation']
    keyword_counts = {}
    
    for keyword in keywords:
        count = df['title'].str.lower().str.contains(keyword).sum()
        keyword_counts[keyword] = count
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(keyword_counts.keys(), keyword_counts.values(), color='lightgreen', edgecolor='darkgreen')
    plt.title('Research Focus Areas (Keyword Frequency in Titles)')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Abstract length analysis
    st.subheader("Abstract Length Distribution")
    df['abstract_length'] = df['abstract'].str.len()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(df['abstract_length'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Abstract Lengths')
    plt.xlabel('Abstract Length (characters)')
    plt.ylabel('Frequency')
    st.pyplot(fig)

def show_research_insights(df):
    st.header("💡 Research Insights")
    
    st.subheader("Key Findings")
    
    insights = [
        "📚 **Research Diversity**: The dataset covers multiple aspects of COVID-19 including clinical, epidemiological, and social perspectives",
        "⏰ **Timely Publications**: Most research was published during the early months of the pandemic (March-April 2020)",
        "🔬 **Multi-disciplinary Approach**: Papers span various fields from medicine to economics and psychology",
        "📄 **Data Availability**: Majority of papers have associated PDF or XML content for detailed analysis",
        "👥 **Collaborative Research**: Multiple authors per paper indicate collaborative research efforts"
    ]
    
    for insight in insights:
        st.write(insight)

if __name__ == "__main__":
    main()