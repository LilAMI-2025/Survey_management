#!/usr/bin/env python3
"""
Enhanced UID Management System - Snowflake Optimized Version
Leverages direct Snowflake queries for maximum performance and data coverage.
Based on successful validation of CHOICE_TEXT column and deduplication logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import text, create_engine
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import core matching functions from uid_promax10
from uid_promax10 import (
    get_enhanced_snowflake_engine, compute_tfidf_matches, compute_semantic_matches,
    finalize_matches, detect_uid_conflicts, convert_uid_to_3_chars,
    enhanced_normalize, clean_question_text, contains_identity_info
)

# Configuration
SNOWFLAKE_QUERY_LIMIT = 10000  # Expanded from 100 for comprehensive data
DEFAULT_SURVEY_STAGES = ["Annual Impact Survey", "AP Survey"]  # Default filter options
CACHE_TTL = 3600  # 1 hour cache

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= CORE SNOWFLAKE FUNCTIONS =============

@st.cache_data(ttl=CACHE_TTL)
def get_comprehensive_question_bank_from_snowflake(
    limit=SNOWFLAKE_QUERY_LIMIT, 
    survey_stages=None,
    question_families=None,
    include_choices=True
):
    """
    Get comprehensive question bank directly from Snowflake with deduplication.
    
    Args:
        limit: Maximum number of records to retrieve
        survey_stages: List of survey stages to filter by (None for all)
        question_families: List of question families to filter by (None for all)
        include_choices: Whether to include choice text in results
    
    Returns:
        DataFrame with deduplicated questions and choices
    """
    
    try:
        engine = get_enhanced_snowflake_engine()
        
        # Build WHERE conditions
        where_conditions = [
            "UID IS NOT NULL",
            "HEADING_0 IS NOT NULL", 
            "TRIM(HEADING_0) != ''"
        ]
        
        if survey_stages:
            stages_str = "', '".join(survey_stages)
            where_conditions.append(f"SURVEY_STAGE IN ('{stages_str}')")
            
        if question_families:
            families_str = "', '".join(question_families)
            where_conditions.append(f"QUESTION_FAMILY IN ('{families_str}')")
        
        where_clause = " AND ".join(where_conditions)
        
        # Main query with deduplication
        query = text(f"""
        WITH ranked_data AS (
            SELECT 
                HEADING_0 as question_text,
                {'CHOICE_TEXT as choice_text,' if include_choices else 'NULL as choice_text,'}
                UID,
                QUESTION_FAMILY,
                SURVEY_STAGE,
                SURVEY_ID,
                DATE_CREATED,
                DATE_MODIFIED,
                COUNT(*) as usage_count,
                COUNT(DISTINCT SURVEY_ID) as survey_count,
                ROW_NUMBER() OVER (
                    PARTITION BY HEADING_0{', COALESCE(CHOICE_TEXT, \'\')' if include_choices else ''} 
                    ORDER BY DATE_MODIFIED DESC, DATE_CREATED DESC, COUNT(*) DESC
                ) as recency_rank
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE {where_clause}
            GROUP BY HEADING_0{', CHOICE_TEXT' if include_choices else ''}, UID, QUESTION_FAMILY, SURVEY_STAGE, SURVEY_ID, DATE_CREATED, DATE_MODIFIED
        ),
        final_data AS (
            SELECT 
                question_text,
                choice_text,
                UID as most_recent_uid,
                QUESTION_FAMILY,
                SURVEY_STAGE,
                usage_count,
                survey_count,
                DATE_MODIFIED as most_recent_date,
                CASE 
                    WHEN choice_text IS NULL OR choice_text = '' THEN 'Main Question'
                    ELSE 'Choice'
                END as question_type
            FROM ranked_data 
            WHERE recency_rank = 1
        )
        SELECT * FROM final_data
        ORDER BY SURVEY_STAGE, usage_count DESC, question_type, question_text
        LIMIT {limit}
        """)
        
        with engine.connect() as conn:
            result_df = pd.read_sql(query, conn)
        
        logger.info(f"✅ Retrieved {len(result_df)} deduplicated question-choice combinations from Snowflake")
        
        # Convert UIDs to 3-character format
        if 'MOST_RECENT_UID' in result_df.columns:
            result_df['MOST_RECENT_UID'] = result_df['MOST_RECENT_UID'].apply(
                lambda uid: convert_uid_to_3_chars(uid) if pd.notna(uid) else uid
            )
        
        # Add normalized text for matching
        result_df['normalized_text'] = result_df['QUESTION_TEXT'].apply(enhanced_normalize)
        
        # Mark identity questions
        result_df['is_identity'] = result_df['QUESTION_TEXT'].apply(contains_identity_info)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to get question bank from Snowflake: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def get_survey_stage_options():
    """Get available survey stages from Snowflake for filtering"""
    try:
        engine = get_enhanced_snowflake_engine()
        query = text("""
        SELECT DISTINCT SURVEY_STAGE, COUNT(*) as question_count
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL AND SURVEY_STAGE IS NOT NULL
        GROUP BY SURVEY_STAGE
        ORDER BY question_count DESC
        """)
        
        with engine.connect() as conn:
            result_df = pd.read_sql(query, conn)
        
        return result_df['SURVEY_STAGE'].tolist()
        
    except Exception as e:
        logger.error(f"Failed to get survey stage options: {e}")
        return DEFAULT_SURVEY_STAGES

@st.cache_data(ttl=CACHE_TTL)
def get_question_family_options():
    """Get available question families from Snowflake for filtering"""
    try:
        engine = get_enhanced_snowflake_engine()
        query = text("""
        SELECT DISTINCT QUESTION_FAMILY, COUNT(*) as question_count
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE UID IS NOT NULL AND QUESTION_FAMILY IS NOT NULL
        GROUP BY QUESTION_FAMILY
        ORDER BY question_count DESC
        """)
        
        with engine.connect() as conn:
            result_df = pd.read_sql(query, conn)
        
        return result_df['QUESTION_FAMILY'].tolist()
        
    except Exception as e:
        logger.error(f"Failed to get question family options: {e}")
        return ["matrix", "open_ended", "single_choice", "multiple_choice"]

@st.cache_data(ttl=CACHE_TTL)
def get_snowflake_analytics():
    """Get analytics data about the Snowflake question bank"""
    try:
        engine = get_enhanced_snowflake_engine()
        query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT HEADING_0) as unique_questions,
            COUNT(DISTINCT CHOICE_TEXT) as unique_choices,
            COUNT(DISTINCT UID) as unique_uids,
            COUNT(DISTINCT SURVEY_STAGE) as survey_stages,
            COUNT(DISTINCT QUESTION_FAMILY) as question_families,
            COUNT(DISTINCT SURVEY_ID) as unique_surveys,
            AVG(CASE WHEN UID IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100 as uid_coverage_percent
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE HEADING_0 IS NOT NULL AND TRIM(HEADING_0) != ''
        """)
        
        with engine.connect() as conn:
            result_df = pd.read_sql(query, conn)
        
        return result_df.iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get Snowflake analytics: {e}")
        return {}

# ============= ENHANCED MATCHING FUNCTIONS =============

def run_snowflake_optimized_matching(question_bank_df, target_questions_df):
    """
    Run optimized UID matching using Snowflake question bank.
    Integrates with existing matching logic from uid_promax10.py
    """
    
    if question_bank_df.empty or target_questions_df.empty:
        st.error("Input data is empty for matching.")
        return pd.DataFrame()
    
    try:
        logger.info("🚀 Starting Snowflake-optimized UID matching...")
        
        # Prepare question bank for matching (use same structure as uid_promax10)
        question_bank_for_matching = question_bank_df.copy()
        question_bank_for_matching['HEADING_0'] = question_bank_for_matching['QUESTION_TEXT']
        question_bank_for_matching['UID'] = question_bank_for_matching['MOST_RECENT_UID']
        
        # Run TF-IDF matching
        with st.spinner("Computing TF-IDF matches..."):
            target_with_tfidf = compute_tfidf_matches(
                question_bank_for_matching, 
                target_questions_df
            )
        
        # Run semantic matching
        with st.spinner("Computing semantic matches..."):
            target_with_semantic = compute_semantic_matches(
                question_bank_for_matching, 
                target_with_tfidf
            )
        
        # Finalize matches
        with st.spinner("Finalizing matches..."):
            final_results = finalize_matches(target_with_semantic, question_bank_for_matching)
        
        # Detect conflicts
        final_results = detect_uid_conflicts(final_results)
        
        logger.info(f"✅ Snowflake-optimized matching completed: {len(final_results)} questions processed")
        return final_results
        
    except Exception as e:
        logger.error(f"Snowflake-optimized matching failed: {e}")
        return target_questions_df

# ============= STREAMLIT UI FUNCTIONS =============

def show_snowflake_question_bank_builder():
    """Enhanced question bank builder using Snowflake"""
    
    st.header("🏗️ Snowflake-Optimized Question Bank Builder")
    st.markdown("*Build comprehensive question banks directly from Snowflake data*")
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("🎛️ Filters")
        
        # Survey stage filter
        available_stages = get_survey_stage_options()
        selected_stages = st.multiselect(
            "Survey Stages",
            available_stages,
            default=available_stages[:2] if len(available_stages) >= 2 else available_stages,
            help="Select survey stages to include"
        )
        
        # Question family filter
        available_families = get_question_family_options()
        selected_families = st.multiselect(
            "Question Families",
            available_families,
            default=available_families,
            help="Select question families to include"
        )
        
        # Other options
        include_choices = st.checkbox("Include Choice Text", value=True)
        query_limit = st.number_input("Record Limit", min_value=100, max_value=50000, value=10000, step=1000)
        
        # Build button
        build_bank = st.button("🚀 Build Question Bank", type="primary")
    
    # Main content
    if build_bank:
        with st.spinner("Building comprehensive question bank from Snowflake..."):
            question_bank_df = get_comprehensive_question_bank_from_snowflake(
                limit=query_limit,
                survey_stages=selected_stages if selected_stages else None,
                question_families=selected_families if selected_families else None,
                include_choices=include_choices
            )
        
        if not question_bank_df.empty:
            # Store in session state
            st.session_state.snowflake_question_bank = question_bank_df
            
            # Display summary
            st.success(f"✅ Built question bank with {len(question_bank_df):,} records")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                main_questions = len(question_bank_df[question_bank_df['QUESTION_TYPE'] == 'Main Question'])
                st.metric("Main Questions", f"{main_questions:,}")
            
            with col2:
                choices = len(question_bank_df[question_bank_df['QUESTION_TYPE'] == 'Choice'])
                st.metric("Choices", f"{choices:,}")
            
            with col3:
                unique_uids = question_bank_df['MOST_RECENT_UID'].nunique()
                st.metric("Unique UIDs", unique_uids)
            
            with col4:
                unique_stages = question_bank_df['SURVEY_STAGE'].nunique()
                st.metric("Survey Stages", unique_stages)
            
            # Display breakdown by survey stage
            st.subheader("📊 Breakdown by Survey Stage")
            stage_summary = question_bank_df.groupby('SURVEY_STAGE').agg({
                'QUESTION_TEXT': 'count',
                'MOST_RECENT_UID': 'nunique',
                'USAGE_COUNT': 'sum',
                'SURVEY_COUNT': 'sum'
            }).round().astype(int)
            stage_summary.columns = ['Total Records', 'Unique UIDs', 'Total Usage', 'Total Surveys']
            st.dataframe(stage_summary, use_container_width=True)
            
            # Sample data preview
            st.subheader("🔍 Sample Data Preview")
            preview_cols = ['QUESTION_TEXT', 'CHOICE_TEXT', 'MOST_RECENT_UID', 'QUESTION_FAMILY', 'SURVEY_STAGE', 'USAGE_COUNT']
            st.dataframe(question_bank_df[preview_cols].head(20), use_container_width=True)
            
        else:
            st.error("❌ Failed to build question bank. Please check your filters and try again.")
    
    # Analytics section
    st.subheader("📈 Snowflake Data Analytics")
    
    if st.button("📊 Load Analytics"):
        with st.spinner("Loading Snowflake analytics..."):
            analytics = get_snowflake_analytics()
        
        if analytics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{analytics.get('TOTAL_RECORDS', 0):,}")
                st.metric("Unique Questions", f"{analytics.get('UNIQUE_QUESTIONS', 0):,}")
            
            with col2:
                st.metric("Unique Choices", f"{analytics.get('UNIQUE_CHOICES', 0):,}")
                st.metric("Unique UIDs", analytics.get('UNIQUE_UIDS', 0))
            
            with col3:
                st.metric("Survey Stages", analytics.get('SURVEY_STAGES', 0))
                st.metric("UID Coverage", f"{analytics.get('UID_COVERAGE_PERCENT', 0):.1f}%")

def show_snowflake_matching():
    """Enhanced matching interface using Snowflake question bank"""
    
    st.header("🎯 Snowflake-Optimized UID Matching")
    st.markdown("*Match target questions against comprehensive Snowflake question bank*")
    
    # Check if question bank is available
    if 'snowflake_question_bank' not in st.session_state:
        st.warning("⚠️ No Snowflake question bank loaded. Please build one first.")
        if st.button("🏗️ Go to Question Bank Builder"):
            st.switch_page("snowflake_question_bank_builder")
        return
    
    question_bank_df = st.session_state.snowflake_question_bank
    st.info(f"📊 Using question bank with {len(question_bank_df):,} records")
    
    # File upload for target questions
    st.subheader("📁 Upload Target Questions")
    uploaded_file = st.file_uploader(
        "Choose CSV file with target questions",
        type=['csv'],
        help="CSV should have a 'question_text' column"
    )
    
    if uploaded_file:
        try:
            target_df = pd.read_csv(uploaded_file)
            
            if 'question_text' not in target_df.columns:
                st.error("❌ CSV must contain a 'question_text' column")
                return
            
            st.success(f"✅ Loaded {len(target_df)} target questions")
            st.dataframe(target_df.head(), use_container_width=True)
            
            # Run matching
            if st.button("🚀 Start Matching", type="primary"):
                with st.spinner("Running Snowflake-optimized matching..."):
                    matched_results = run_snowflake_optimized_matching(question_bank_df, target_df)
                
                if not matched_results.empty:
                    st.session_state.matching_results = matched_results
                    
                    # Display results summary
                    st.success(f"✅ Matching completed for {len(matched_results)} questions")
                    
                    # Matching statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        high_conf = len(matched_results[matched_results.get('Final_Match_Type', '').str.contains('High', na=False)])
                        st.metric("High Confidence", high_conf)
                    
                    with col2:
                        uid_final = len(matched_results[matched_results.get('Final_Match_Type', '').str.contains('UID Final', na=False)])
                        st.metric("UID Final Matches", uid_final)
                    
                    with col3:
                        semantic = len(matched_results[matched_results.get('Final_Match_Type', '').str.contains('Semantic', na=False)])
                        st.metric("Semantic Matches", semantic)
                    
                    with col4:
                        no_match = len(matched_results[matched_results.get('Final_Match_Type', '').str.contains('No match', na=False)])
                        st.metric("No Match", no_match)
                    
                    # Results preview
                    st.subheader("🎯 Matching Results Preview")
                    results_cols = ['question_text', 'Final_UID', 'Final_Match_Type', 'Similarity']
                    if 'Final_UID' in matched_results.columns:
                        st.dataframe(matched_results[results_cols].head(20), use_container_width=True)
                    
                else:
                    st.error("❌ Matching failed")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def show_snowflake_export():
    """Export functionality for Snowflake results"""
    
    st.header("📤 Export Snowflake Results")
    st.markdown("*Export question banks and matching results*")
    
    # Question bank export
    if 'snowflake_question_bank' in st.session_state:
        st.subheader("📊 Question Bank Export")
        question_bank_df = st.session_state.snowflake_question_bank
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Records", f"{len(question_bank_df):,}")
        with col2:
            csv_data = question_bank_df.to_csv(index=False)
            st.download_button(
                "📥 Download Question Bank CSV",
                csv_data,
                f"snowflake_question_bank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    # Matching results export
    if 'matching_results' in st.session_state:
        st.subheader("🎯 Matching Results Export")
        matching_results = st.session_state.matching_results
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Matched Questions", f"{len(matching_results):,}")
        with col2:
            csv_data = matching_results.to_csv(index=False)
            st.download_button(
                "📥 Download Matching Results CSV",
                csv_data,
                f"snowflake_matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

# ============= MAIN APPLICATION =============

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Snowflake UID Management",
        page_icon="❄️",
        layout="wide"
    )
    
    st.title("❄️ Snowflake-Optimized UID Management System")
    st.markdown("*Leveraging direct Snowflake queries for maximum performance and data coverage*")
    
    # Navigation
    tab1, tab2, tab3 = st.tabs([
        "🏗️ Question Bank Builder", 
        "🎯 UID Matching", 
        "📤 Export Results"
    ])
    
    with tab1:
        show_snowflake_question_bank_builder()
    
    with tab2:
        show_snowflake_matching()
    
    with tab3:
        show_snowflake_export()

if __name__ == "__main__":
    main() 