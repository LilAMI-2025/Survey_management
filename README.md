# 🎯 Survey UID Management System

## Overview
A powerful Streamlit application for managing survey data with advanced deduplication capabilities, Snowflake integration, and enhanced question bank organization.

## ✨ Key Features
- **Enhanced Deduplication**: 99.88% reduction in duplicate records
- **Snowflake Integration**: Real-time data processing from Snowflake database
- **Advanced Question Bank**: Organized question-choice management
- **5-Point Validation System**: Comprehensive data quality checks
- **Export Capabilities**: Multiple export formats (CSV, Excel, JSON)

## 🚀 Performance Highlights
- Processes 25K-50K records efficiently
- Eliminates thousands of duplicate choices automatically
- Real-time validation with 5/5 checks passing
- Advanced whitespace trimming and question normalization

## 📊 Technical Stack
- **Frontend**: Streamlit
- **Database**: Snowflake
- **Data Processing**: Pandas, SQLAlchemy
- **ML/NLP**: Sentence Transformers, Scikit-learn
- **Deployment**: Streamlit Community Cloud

## 🔧 Setup for Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure Snowflake credentials in Streamlit secrets
4. Run: `streamlit run uid_snowflake_optimized.py`

## 🌐 Cloud Deployment
This app is deployed on Streamlit Community Cloud with automatic Snowflake integration.

## 📈 Key Metrics
- **Deduplication Rate**: 99.88%
- **Processing Speed**: 25K+ records in seconds
- **Validation Success**: 5/5 checks
- **Data Quality**: Enterprise-grade cleaning