#!/usr/bin/env python3
"""
Enhanced UID Management System - Snowflake Optimized Version
Leverages direct Snowflake queries for maximum performance and data coverage.
Based on successful validation of CHOICE_TEXT column and deduplication logic.
Complete standalone implementation without external dependencies.
"""

import streamlit as st
import requests

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Snowflake UID Management",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="❄️"
)

import pandas as pd
import numpy as np
from sqlalchemy import text, create_engine
import logging
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import warnings
warnings.filterwarnings('ignore')
import io

# Configuration
SNOWFLAKE_QUERY_LIMIT = 50000
CACHE_TTL = 300  # 5 minutes
BATCH_SIZE = 100
TFIDF_HIGH_CONFIDENCE = 0.7
TFIDF_LOW_CONFIDENCE = 0.4
SEMANTIC_THRESHOLD = 0.75

# Enhanced synonym mapping for text normalization
ENHANCED_SYNONYM_MAP = {
    'organisation': 'organization', 'organisations': 'organizations',
    'colour': 'color', 'colours': 'colors', 'realise': 'realize', 'realised': 'realized',
    'utilise': 'utilize', 'utilised': 'utilized', 'favourite': 'favorite',
    'centre': 'center', 'theatre': 'theater', 'metre': 'meter',
    'programme': 'program', 'programmes': 'programs',
    'behaviour': 'behavior', 'behaviours': 'behaviors',
    'honour': 'honor', 'honours': 'honors', 'labour': 'labor',
    'neighbour': 'neighbor', 'neighbours': 'neighbors',
    'analyse': 'analyze', 'analysed': 'analyzed', 'analyses': 'analyzes',
    'capitalise': 'capitalize', 'capitalised': 'capitalized',
    'categorise': 'categorize', 'categorised': 'categorized',
    'emphasise': 'emphasize', 'emphasised': 'emphasized',
    'minimise': 'minimize', 'minimised': 'minimized',
    'optimise': 'optimize', 'optimised': 'optimized',
    'recognise': 'recognize', 'recognised': 'recognized',
    'summarise': 'summarize', 'summarised': 'summarized',
    'synthesise': 'synthesize', 'synthesised': 'synthesized',
    'whilst': 'while', 'amongst': 'among', 'towards': 'toward'
}

# Identity detection patterns
IDENTITY_PATTERNS = {
    'Age': [r'\bage\b', r'\bold\b', r'\byoung\b', r'\byears?\s+old\b', r'\bbirth\s*year\b', r'\bdob\b', r'\bdate\s+of\s+birth\b'],
    'City': [r'\bcity\b', r'\btown\b', r'\bmunicipality\b', r'\burban\s+area\b', r'\blocal\s+area\b'],
    'Company': [r'\bcompany\b', r'\borganiz[as]tion\b', r'\bbusiness\b', r'\bemployer\b', r'\bworkplace\b', r'\bfirm\b', r'\bcorporation\b'],
    'Country': [r'\bcountry\b', r'\bnation\b', r'\bnationality\b', r'\bcitizenship\b', r'\bpassport\b'],
    'Date of Birth': [r'\bdate\s+of\s+birth\b', r'\bdob\b', r'\bbirth\s*date\b', r'\bbirthday\b', r'\bborn\s+on\b'],
    'E-Mail': [r'\bemail\b', r'\be-mail\b', r'\belectronic\s+mail\b', r'@', r'\bcontact\s+details\b'],
    'Education level': [r'\beducation\b', r'\bdegree\b', r'\bqualification\b', r'\buniversity\b', r'\bcollege\b', r'\bschool\b', r'\bacademic\b'],
    'Full Name': [r'\bfull\s+name\b', r'\bcomplete\s+name\b', r'\bfirst\s+and\s+last\s+name\b'],
    'First Name': [r'\bfirst\s+name\b', r'\bgiven\s+name\b', r'\bforename\b'],
    'Last Name': [r'\blast\s+name\b', r'\bsurname\b', r'\bfamily\s+name\b'],
    'Gender': [r'\bgender\b', r'\bsex\b', r'\bmale\b', r'\bfemale\b', r'\bman\b', r'\bwoman\b', r'\bpronoun\b'],
    'Location': [r'\blocation\b', r'\baddress\b', r'\bwhere\s+do\s+you\s+live\b', r'\bresidence\b', r'\bpostal\s+code\b', r'\bzip\s+code\b'],
    'Phone Number': [r'\bphone\b', r'\btelephone\b', r'\bmobile\b', r'\bcell\b', r'\bnumber\b', r'\bcontact\s+number\b'],
    'Region': [r'\bregion\b', r'\bstate\b', r'\bprovince\b', r'\barea\b', r'\bdistrict\b', r'\bcounty\b'],
    'Title/Role': [r'\btitle\b', r'\brole\b', r'\bposition\b', r'\bjob\b', r'\boccupation\b', r'\bprofession\b'],
    'PIN/Passport': [r'\bpin\b', r'\bpassport\b', r'\bid\s+number\b', r'\bidentification\b', r'\bsocial\s+security\b']
}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= SESSION STATE INITIALIZATION =============
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'question_bank_data' not in st.session_state:
        st.session_state.question_bank_data = pd.DataFrame()
    if 'matching_results' not in st.session_state:
        st.session_state.matching_results = pd.DataFrame()
    if 'selected_survey_stages' not in st.session_state:
        st.session_state.selected_survey_stages = []

# Initialize session state
initialize_session_state()

# ============= CORE FUNCTIONS =============
@st.cache_resource
def get_snowflake_engine():
    """Create and return Snowflake engine"""
    try:
        connection_string = (
            f"snowflake://{st.secrets['snowflake']['user']}:"
            f"{st.secrets['snowflake']['password']}@"
            f"{st.secrets['snowflake']['account']}/"
            f"{st.secrets['snowflake']['database']}/"
            f"{st.secrets['snowflake']['schema']}?"
            f"warehouse={st.secrets['snowflake']['warehouse']}&"
            f"role={st.secrets['snowflake']['role']}"
        )
        return create_engine(connection_string)
    except Exception as e:
        logger.error(f"Failed to create Snowflake engine: {e}")
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load SentenceTransformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    """Enhanced text normalization with synonym mapping"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    normalized_words = [synonym_map.get(word, word) for word in words]
    return ' '.join(normalized_words)

def clean_question_text(text):
    """Clean question text for better matching"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    text = re.sub(r'\s*\[.*?\]\s*', '', text)
    text = re.sub(r'\s*\(.*?\)\s*', '', text)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def contains_identity_info(text):
    """Check if text contains identity information"""
    if pd.isna(text) or text == "":
        return False
    
    text_lower = str(text).lower()
    for identity_type, patterns in IDENTITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
    return False

def determine_identity_type(text):
    """Determine the specific type of identity information"""
    if pd.isna(text) or text == "":
        return "Not Identity"
    
    text_lower = str(text).lower()
    
    # Check each identity type
    for identity_type, patterns in IDENTITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return identity_type
    
    return "Not Identity"

def convert_uid_to_3_chars(uid):
    """Convert UID to 3-character format following Google Sheets logic"""
    if pd.isna(uid) or uid == "":
        return uid
    
    uid_str = str(uid).strip()
    if len(uid_str) == 3:
        return uid_str
    elif len(uid_str) < 3:
        return uid_str.zfill(3)
    else:
        return uid_str[:3]

@st.cache_data(ttl=CACHE_TTL)
def get_tfidf_vectors(df_reference):
    """Get TF-IDF vectors for reference questions"""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    
    normalized_texts = df_reference['norm_text'].fillna('').tolist()
    vectors = vectorizer.fit_transform(normalized_texts)
    
    return vectorizer, vectors

# ============= CORE SNOWFLAKE FUNCTIONS =============

@st.cache_data(ttl=CACHE_TTL)
def get_comprehensive_question_bank_from_snowflake(
    limit=SNOWFLAKE_QUERY_LIMIT, 
    survey_stages=None,
    include_choices=True,
    grouped_by_stage=True
):
    """
    Get comprehensive question bank from Snowflake with two modes:
    - grouped_by_stage=True: Questions grouped by survey stage (for client-specific modifications)
    - grouped_by_stage=False: Unique questions across all stages (prevents duplicate UIDs)
    """
    
    try:
        engine = get_snowflake_engine()
        if engine is None:
            return pd.DataFrame()
        
        # Base conditions with CALA filter - remove any questions containing CALA
        base_conditions = """
        WHERE HEADING_0 IS NOT NULL 
        AND TRIM(HEADING_0) != ''
        AND UPPER(HEADING_0) NOT LIKE '%CALA%'
        AND UPPER(COALESCE(CHOICE_TEXT, '')) NOT LIKE '%CALA%'
        """
        
        # Add survey stage filter if specified
        stage_filter = ""
        if survey_stages and len(survey_stages) > 0:
            stages_str = "', '".join(survey_stages)
            stage_filter = f" AND SURVEY_STAGE IN ('{stages_str}')"
        
        if grouped_by_stage:
            # For grouped by stage (grouped_by_stage=True), maintain stage grouping  
            query = f"""
            SELECT 
                HEADING_0 as QUESTION_TEXT,
                COALESCE(CHOICE_TEXT, '') as CHOICE_TEXT,
                UID,
                SURVEY_STAGE,
                COUNT(*) as FREQUENCY
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            {base_conditions}
            GROUP BY HEADING_0, COALESCE(CHOICE_TEXT, ''), UID, SURVEY_STAGE
            ORDER BY UID, FREQUENCY DESC
            LIMIT {limit}
            """
        else:
            # Unique questions across all stages (new behavior)
            if include_choices:
                query = f"""
                WITH ranked_data AS (
                    SELECT 
                        HEADING_0,
                        COALESCE(CHOICE_TEXT, '') as CHOICE_TEXT,
                        UID,
                        MAX(SURVEY_STAGE) as LATEST_STAGE,
                        SUM(1) as TOTAL_FREQUENCY,
                        ROW_NUMBER() OVER (
                            PARTITION BY HEADING_0, COALESCE(CHOICE_TEXT, '') 
                            ORDER BY MAX(DATE_MODIFIED) DESC, SUM(1) DESC
                        ) as rn
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    {base_conditions}{stage_filter}
                    GROUP BY HEADING_0, COALESCE(CHOICE_TEXT, ''), UID
                )
                SELECT 
                    HEADING_0 as QUESTION_TEXT,
                    CASE WHEN CHOICE_TEXT = '' THEN NULL ELSE CHOICE_TEXT END as CHOICE_TEXT,
                    UID,
                    LATEST_STAGE as SURVEY_STAGE,
                    TOTAL_FREQUENCY as FREQUENCY
                FROM ranked_data 
                WHERE rn = 1
                ORDER BY UID, FREQUENCY DESC
                LIMIT {limit}
                """
            else:
                query = f"""
                WITH ranked_data AS (
                    SELECT 
                        HEADING_0,
                        UID,
                        MAX(DATE_MODIFIED) as LATEST_MODIFIED,
                        MAX(SURVEY_STAGE) as LATEST_STAGE,
                        SUM(1) as TOTAL_FREQUENCY,
                        ROW_NUMBER() OVER (
                            PARTITION BY HEADING_0 
                            ORDER BY MAX(DATE_MODIFIED) DESC, SUM(1) DESC
                        ) as rn
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    {base_conditions}{stage_filter}
                    GROUP BY HEADING_0, UID
                )
                SELECT 
                    HEADING_0 as QUESTION_TEXT,
                    UID,
                    LATEST_STAGE as SURVEY_STAGE,
                    TOTAL_FREQUENCY as FREQUENCY,
                    LATEST_MODIFIED as DATE_MODIFIED
                FROM ranked_data 
                WHERE rn = 1
                ORDER BY UID, FREQUENCY DESC
                LIMIT {limit}
                """
        
        with engine.connect() as conn:
            result_df = pd.read_sql(text(query), conn)
        
        if not result_df.empty:
            # Handle column name variations (Snowflake returns lowercase)
            column_mapping = {}
            for col in result_df.columns:
                col_upper = col.upper()
                if col_upper == 'QUESTION_TEXT':
                    column_mapping['QUESTION_TEXT'] = col
                elif col_upper == 'CHOICE_TEXT':
                    column_mapping['CHOICE_TEXT'] = col
                elif col_upper == 'UID':
                    column_mapping['UID'] = col
                elif col_upper == 'SURVEY_STAGE':
                    column_mapping['SURVEY_STAGE'] = col
                elif col_upper == 'FREQUENCY':
                    column_mapping['FREQUENCY'] = col
            
            # Rename columns to standardized uppercase names
            for standard_name, actual_name in column_mapping.items():
                if actual_name in result_df.columns:
                    result_df = result_df.rename(columns={actual_name: standard_name})
            
            # Add identity detection columns
            if 'QUESTION_TEXT' in result_df.columns:
                result_df['IS_IDENTITY'] = result_df['QUESTION_TEXT'].apply(contains_identity_info)
                result_df['IDENTITY_TYPE'] = result_df['QUESTION_TEXT'].apply(determine_identity_type)
            
            # Convert UIDs to 3-character format
            if 'UID' in result_df.columns:
                result_df['UID'] = result_df['UID'].apply(convert_uid_to_3_chars)
            
            logger.info(f"✅ Retrieved {len(result_df)} question-choice combinations from Snowflake")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to get question bank from Snowflake: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def get_survey_stage_options():
    """Get available survey stage options from Snowflake"""
    try:
        engine = get_snowflake_engine()
        if engine is None:
            return []
        
        # Direct query since we know the column name is SURVEY_STAGE
        query = """
        SELECT DISTINCT SURVEY_STAGE, COUNT(*) as count
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE SURVEY_STAGE IS NOT NULL 
        AND TRIM(SURVEY_STAGE) != ''
        AND UPPER(HEADING_0) NOT LIKE '%CALA%'
        GROUP BY SURVEY_STAGE
        ORDER BY count DESC
        """
        
        with engine.connect() as conn:
            result_df = pd.read_sql(text(query), conn)
            
        # Handle lowercase column names returned by Snowflake
        if 'survey_stage' in result_df.columns:
            stages = result_df['survey_stage'].tolist()
        elif 'SURVEY_STAGE' in result_df.columns:
            stages = result_df['SURVEY_STAGE'].tolist()
        else:
            stages = []
            
        if stages:
            logger.info(f"✅ Retrieved {len(stages)} survey stages from Snowflake: {stages}")
        return stages
        
    except Exception as e:
        logger.error(f"Failed to get survey stage options: {str(e)}")
        # Return the known stages as fallback
        return [
            "Annual Impact Survey", "Pre-Programme Survey", "Enrollment/Application Survey",
            "Progress Review Survey", "Other", "Growth Goal Reflection", "Change Challenge Survey",
            "Pulse Check Survey", "LL Feedback Survey", "AP Survey", "CEO/Client Lead Survey",
            "Longitudinal Survey"
        ]

@st.cache_data(ttl=CACHE_TTL)
def get_snowflake_analytics():
    """Get analytics data about the Snowflake question bank"""
    try:
        engine = get_snowflake_engine()
        if engine is None:
            return {}
            
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

@st.cache_data(ttl=CACHE_TTL)
def compute_tfidf_matches(df_reference, df_target):
    """Compute TF-IDF based matches between reference and target questions"""
    # df_reference has QUESTION_TEXT (from Snowflake), df_target has question_text (from user)
    df_reference = df_reference[df_reference["QUESTION_TEXT"].notna()].reset_index(drop=True)
    df_target = df_target[df_target["question_text"].notna()].reset_index(drop=True)
    
    # Create normalized text columns
    df_reference["norm_text"] = df_reference["QUESTION_TEXT"].apply(enhanced_normalize)
    df_target["norm_text"] = df_target["question_text"].apply(enhanced_normalize)

    vectorizer, ref_vectors = get_tfidf_vectors(df_reference)
    target_vectors = vectorizer.transform(df_target["norm_text"])
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)

    matched_uids, matched_qs, scores, confs = [], [], [], []
    for sim_row in similarity_matrix:
        best_idx = sim_row.argmax()
        best_score = sim_row[best_idx]
        if best_score >= TFIDF_HIGH_CONFIDENCE:
            conf = "✅ High"
        elif best_score >= TFIDF_LOW_CONFIDENCE:
            conf = "⚠️ Low"
        else:
            conf = "❌ No match"
            best_idx = None
            
        matched_uids.append(df_reference.iloc[best_idx]["UID"] if best_idx is not None else None)
        matched_qs.append(df_reference.iloc[best_idx]["QUESTION_TEXT"] if best_idx is not None else None)
        scores.append(round(best_score, 4))
        confs.append(conf)

    df_target["Suggested_UID"] = matched_uids
    df_target["Matched_Question"] = matched_qs
    df_target["Similarity"] = scores
    df_target["Match_Confidence"] = confs
    return df_target

def compute_semantic_matches(df_reference, df_target):
    """Compute semantic similarity matches using SentenceTransformer"""
    model = load_sentence_transformer()
    
    # df_target has question_text, df_reference has QUESTION_TEXT
    emb_target = model.encode(df_target["question_text"].tolist(), convert_to_tensor=True)
    emb_ref = model.encode(df_reference["QUESTION_TEXT"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_target, emb_ref)

    sem_matches, sem_scores = [], []
    for i in range(len(df_target)):
        best_idx = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_idx].item()
        sem_matches.append(df_reference.iloc[best_idx]["UID"] if score >= SEMANTIC_THRESHOLD else None)
        sem_scores.append(round(score, 4) if score >= SEMANTIC_THRESHOLD else None)

    df_target["Semantic_UID"] = sem_matches
    df_target["Semantic_Similarity"] = sem_scores
    return df_target

def finalize_matches(df_target, df_reference):
    """Finalize UID matches and create final results"""
    
    # Prioritize TF-IDF, then semantic
    df_target["Final_UID"] = (df_target.get("Suggested_UID", pd.Series(dtype='object'))
                              .combine_first(df_target.get("Semantic_UID", pd.Series(dtype='object'))))
    
    # Convert UIDs to 3 character format
    uid_columns = ['Final_UID', 'Suggested_UID', 'Semantic_UID']
    for col in uid_columns:
        if col in df_target.columns:
            df_target[col] = df_target[col].apply(lambda uid: convert_uid_to_3_chars(uid) if pd.notna(uid) else uid)
    
    # Check for identity types and remove UID assignments
    if 'question_text' in df_target.columns:
        identity_mask = df_target['question_text'].apply(contains_identity_info)
        df_target.loc[identity_mask, ['Final_UID', 'Suggested_UID', 'Semantic_UID']] = None
        logger.info(f"🔐 Removed UID assignments from {identity_mask.sum()} identity questions")
    
    df_target["configured_final_UID"] = df_target["Final_UID"]
    df_target["Final_Question"] = df_target.get("Matched_Question", "")
    
    # Update Final_Match_Type to reflect priority
    def get_final_match_type(row):
        # Check if it's an identity type first
        if pd.notna(row.get('question_text')) and contains_identity_info(row['question_text']):
            return "🔐 Identity (No UID)"
        elif pd.notnull(row.get("Suggested_UID")):
            return row.get("Match_Confidence", "✅ High")
        elif pd.notnull(row.get("Semantic_UID")):
            return "🧠 Semantic"
        else:
            return "❌ No match"
    
    df_target["Final_Match_Type"] = df_target.apply(get_final_match_type, axis=1)
    
    return df_target

def detect_uid_conflicts(df_target):
    """Detect and mark UID conflicts"""
    # Simple conflict detection - mark duplicates
    if 'Final_UID' in df_target.columns:
        df_target['has_conflict'] = df_target['Final_UID'].duplicated(keep=False) & df_target['Final_UID'].notna()
    else:
        df_target['has_conflict'] = False
    
    return df_target

def run_snowflake_optimized_matching(question_bank_df, target_questions_df):
    """
    Run optimized UID matching using Snowflake question bank.
    Complete standalone implementation.
    """
    
    if question_bank_df.empty or target_questions_df.empty:
        st.error("Input data is empty for matching.")
        return pd.DataFrame()
    
    try:
        logger.info("🚀 Starting Snowflake-optimized UID matching...")
        
        # Ensure target dataframe has the expected column
        if 'question_text' not in target_questions_df.columns:
            st.error("❌ Target dataframe must have 'question_text' column")
            return pd.DataFrame()
        
        # Run TF-IDF matching
        with st.spinner("Computing TF-IDF matches..."):
            target_with_tfidf = compute_tfidf_matches(question_bank_df, target_questions_df)
        
        # Run semantic matching
        with st.spinner("Computing semantic matches..."):
            target_with_semantic = compute_semantic_matches(question_bank_df, target_with_tfidf)
        
        # Finalize matches
        with st.spinner("Finalizing matches..."):
            final_results = finalize_matches(target_with_semantic, question_bank_df)
        
        # Detect conflicts
        final_results = detect_uid_conflicts(final_results)
        
        logger.info(f"✅ Snowflake-optimized matching completed: {len(final_results)} questions processed")
        return final_results
        
    except Exception as e:
        logger.error(f"Snowflake-optimized matching failed: {e}")
        st.error(f"❌ Matching failed: {str(e)}")
        return target_questions_df

# ============= STREAMLIT UI FUNCTIONS =============

def show_grouped_question_bank_builder():
    """Show question bank builder grouped by survey stages"""
    st.header("🏗️ Question Bank Builder (Grouped by Survey Stages)")
    st.markdown("*Build question banks organized by survey stages for client-specific modifications*")
    
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Snowflake - Survey responses grouped by survey stage</div>', unsafe_allow_html=True)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        # Survey stage selection 
        available_stages = get_survey_stage_options()
        if available_stages:
            selected_stages = st.multiselect(
                "📊 Select Survey Stages",
                available_stages,
                default=available_stages[:3] if len(available_stages) >= 3 else available_stages,
                help="Choose which survey stages to include in the question bank"
            )
        else:
            st.error("❌ No survey stages available")
            return
    
    with col2:
        # Advanced options
        st.subheader("⚙️ Options")
        limit = st.number_input("🔢 Record Limit", min_value=1000, max_value=50000, value=25000)
        include_choices = st.checkbox("📝 Include Choice Text", value=True)
    
    # Build button
    if st.button("🚀 Build Grouped Question Bank", type="primary", key="build_grouped"):
        if not selected_stages:
            st.error("❌ Please select at least one survey stage")
            return
        
        with st.spinner("🔄 Building grouped question bank from Snowflake..."):
            # Force grouped mode
            question_bank_df = get_comprehensive_question_bank_from_snowflake(
                limit=limit,
                survey_stages=selected_stages,
                include_choices=include_choices,
                grouped_by_stage=True  # Force grouped mode
            )
            
            if question_bank_df.empty:
                st.error("❌ Failed to build question bank. Please check your filters and try again.")
                return
            
            # Store in session state
            st.session_state.question_bank_grouped = question_bank_df
            st.session_state.question_bank_type = "Grouped by Survey Stage"
            
            # Display overview
            st.success(f"✅ Built grouped question bank with {len(question_bank_df):,} records")
            
            # Data preview grouped by survey stage
            st.subheader("📊 Question Bank Preview")
            
            # Get correct column names (handle case variations)
            cols = question_bank_df.columns.str.lower().tolist()
            question_col = None
            choice_col = None
            uid_col = None
            stage_col = None
            
            for col in question_bank_df.columns:
                col_lower = col.lower()
                if 'question' in col_lower or 'heading' in col_lower:
                    question_col = col
                elif 'choice' in col_lower:
                    choice_col = col
                elif col_lower == 'uid':
                    uid_col = col
                elif 'stage' in col_lower:
                    stage_col = col
            
            if stage_col and question_col:
                # Group by survey stage
                for stage in question_bank_df[stage_col].unique():
                    stage_data = question_bank_df[question_bank_df[stage_col] == stage]
                    
                    with st.expander(f"📋 {stage} ({len(stage_data)} questions)"):
                        # Show columns that exist
                        display_cols = []
                        if question_col: display_cols.append(question_col)
                        if choice_col: display_cols.append(choice_col)
                        if uid_col: display_cols.append(uid_col)
                        
                        if display_cols:
                            st.dataframe(
                                stage_data[display_cols].head(20).fillna(''),
                                use_container_width=True
                            )
                        else:
                            st.dataframe(stage_data.head(20), use_container_width=True)
            else:
                # Fallback display
                st.dataframe(question_bank_df.head(100), use_container_width=True)

def show_unique_question_bank_builder():
    """Show question bank builder for unique questions across all stages"""
    st.header("🔍 Question Bank Builder (Unique Questions)")
    st.markdown("*Build question banks with unique questions across all stages - prevents duplicate UIDs*")
    
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Snowflake - Deduplicated questions showing most recent UID assignment</div>', unsafe_allow_html=True)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        # Survey stage selection (optional for filtering)
        available_stages = get_survey_stage_options()
        if available_stages:
            selected_stages = st.multiselect(
                "📊 Filter by Survey Stages (Optional)",
                available_stages,
                default=[],
                help="Optional: Filter questions by specific stages, or leave empty for all stages"
            )
            
            if not selected_stages:
                selected_stages = available_stages  # Use all if none selected
        else:
            st.error("❌ No survey stages available")
            return
    
    with col2:
        # Advanced options
        st.subheader("⚙️ Options")
        limit = st.number_input("🔢 Record Limit", min_value=1000, max_value=50000, value=25000, key="unique_limit")
        include_choices = st.checkbox("📝 Include Choice Text", value=True, key="unique_choices")
    
    # Build button
    if st.button("🚀 Build Unique Question Bank", type="primary", key="build_unique"):
        with st.spinner("🔄 Building unique question bank from Snowflake..."):
            # Force unique mode
            question_bank_df = get_comprehensive_question_bank_from_snowflake(
                limit=limit,
                survey_stages=selected_stages,
                include_choices=include_choices,
                grouped_by_stage=False  # Force unique mode
            )
            
            if question_bank_df.empty:
                st.error("❌ Failed to build question bank. Please check your filters and try again.")
                return
            
            # Store in session state
            st.session_state.question_bank_unique = question_bank_df
            st.session_state.question_bank_type = "Unique Questions (All Stages)"
            
            # Display overview
            st.success(f"✅ Built unique question bank with {len(question_bank_df):,} records")
            
            # Data preview - flat list without grouping
            st.subheader("🔍 Unique Questions Preview")
            
            # Get correct column names (handle case variations)
            question_col = None
            choice_col = None
            uid_col = None
            stage_col = None
            freq_col = None
            
            for col in question_bank_df.columns:
                col_lower = col.lower()
                if 'question' in col_lower or 'heading' in col_lower:
                    question_col = col
                elif 'choice' in col_lower:
                    choice_col = col
                elif col_lower == 'uid':
                    uid_col = col
                elif 'stage' in col_lower:
                    stage_col = col
                elif 'freq' in col_lower:
                    freq_col = col
            
            # Show sample of questions in flat format
            display_cols = []
            if question_col: display_cols.append(question_col)
            if choice_col: display_cols.append(choice_col)
            if uid_col: display_cols.append(uid_col)
            if stage_col: display_cols.append(stage_col)
            if freq_col: display_cols.append(freq_col)
            
            if display_cols:
                preview_df = question_bank_df[display_cols].head(100).fillna('')
            else:
                preview_df = question_bank_df.head(100).fillna('')
            
            # Add search functionality
            search_term = st.text_input("🔍 Search questions:", key="search_unique_questions")
            if search_term and question_col:
                mask = preview_df[question_col].astype(str).str.contains(search_term, case=False, na=False)
                preview_df = preview_df[mask]
            
            st.dataframe(
                preview_df,
                use_container_width=True,
                height=400
            )
            
            # Show summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(question_bank_df))
            with col2:
                if question_col:
                    unique_questions = question_bank_df[question_col].nunique()
                    st.metric("Unique Questions", unique_questions)
            with col3:
                if stage_col:
                    stage_count = question_bank_df[stage_col].nunique()
                    st.metric("Survey Stages", stage_count)
            with col4:
                if uid_col:
                    uid_count = question_bank_df[uid_col].nunique()
                    st.metric("Unique UIDs", uid_count)

# ============= MAIN APPLICATION =============
def main():
    """Main Streamlit application with sidebar navigation"""
    
    st.title("❄️ Snowflake UID Management System")
    
    # Sidebar navigation - organized by workflow
    with st.sidebar:
        st.header("🧭 Workflow Navigation")
        st.markdown("*Follow the workflow steps in order:*")
        
        # Handle query parameter navigation
        query_page = st.query_params.get("page", "")
        if query_page == "step1a":
            default_index = 1
        elif query_page == "step1b":
            default_index = 2
        elif query_page == "step2":
            default_index = 3
        elif query_page == "step3":
            default_index = 4
        elif query_page == "step4":
            default_index = 5
        else:
            default_index = 0
        
        page = st.radio(
            "Select workflow step:",
            [
                "🏠 Home Dashboard", 
                "**Step 1A:** 🏗️ Question Bank (Grouped by Stage)",
                "**Step 1B:** 🔍 Question Bank (Unique Questions)",
                "**Step 2:** 📊 Survey Selection", 
                "**Step 3:** 🎯 UID Matching",
                "**Step 4:** 📤 Export Results"
            ],
            index=default_index,
            key="main_navigation"
        )
        
        # Workflow guidance
        st.markdown("---")
        st.markdown("### 📋 Workflow Guide")
        st.markdown("""
        **1. Question Bank Builder**  
        Build comprehensive question banks from Snowflake data
        
        **2. Survey Selection**  
        Select SurveyMonkey surveys to extract questions
        
        **3. UID Matching**  
        Match survey questions to question bank UIDs
        
        **4. Export Results**  
        Export matched data with identity classification
        """)
    
    # Route to appropriate page
    if page == "🏠 Home Dashboard":
        show_home_dashboard()
    elif page == "**Step 1A:** 🏗️ Question Bank (Grouped by Stage)":
        show_grouped_question_bank_builder()
    elif page == "**Step 1B:** 🔍 Question Bank (Unique Questions)":
        show_unique_question_bank_builder()
    elif page == "**Step 2:** 📊 Survey Selection":
        show_survey_selection()
    elif page == "**Step 3:** 🎯 UID Matching":
        show_snowflake_matching()
    elif page == "**Step 4:** 📤 Export Results":
        show_snowflake_export()

def show_home_dashboard():
    """Show enhanced home dashboard with workflow guidance"""
    st.header("🏠 Home Dashboard")
    st.markdown("*Welcome to the Snowflake UID Management System*")
    
    # System status
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            stages = get_survey_stage_options()
            st.success(f"✅ Snowflake Connected - {len(stages)} survey stages available")
        except:
            st.error("❌ Snowflake Connection Failed")
    
    with col2:
        try:
            headers = {"Authorization": f"Bearer {st.secrets['surveymonkey']['access_token']}"}
            st.success("✅ SurveyMonkey Connected")
        except:
            st.error("❌ SurveyMonkey Connection Failed")
    
    st.markdown("---")
    
    # Workflow steps
    st.markdown("### 🚀 Quick Start Workflow")
    
    # Step 1A: Grouped Question Bank
    with st.expander("**Step 1A: 🏗️ Grouped Question Bank**"):
        st.markdown("""
        **Purpose:** Build question banks organized by survey stages
        
        **What you'll do:**
        - Select specific survey stages to include
        - Build question banks grouped by stage for client-specific modifications
        - Questions retain their survey stage context
        - Ideal for stage-specific analysis and modifications
        
        **Output:** Question bank with stage groupings
        """)
        
        if st.button("🚀 Start Step 1A: Grouped Question Bank", type="primary", key="step1a"):
            st.query_params["page"] = "step1a"
            st.rerun()
    
    # Step 1B: Unique Question Bank  
    with st.expander("**Step 1B: 🔍 Unique Question Bank**"):
        st.markdown("""
        **Purpose:** Build deduplicated question banks across all stages
        
        **What you'll do:**
        - Extract unique questions from all survey stages
        - Prevent duplicate UIDs across different stages
        - Shows most recent UID assignment for each question
        - Ideal for comprehensive UID management
        
        **Output:** Flat list of unique questions with latest UIDs
        """)
        
        if st.button("🚀 Start Step 1B: Unique Question Bank", type="primary", key="step1b"):
            st.query_params["page"] = "step1b"
            st.rerun()
    
    # Step 2: Survey Selection
    with st.expander("**Step 2: 📊 Survey Selection**"):
        st.markdown("""
        **Purpose:** Select SurveyMonkey surveys to extract questions from
        
        **What you'll do:**
        - Browse available SurveyMonkey surveys
        - Select specific surveys for question extraction
        - Extract questions and choices from selected surveys
        - Review extracted question data
        
        **Prerequisites:** Complete Step 1 (Question Bank)
        **Output:** Target questions for UID matching
        """)
        
        if st.button("➡️ Start Step 2: Select Surveys", key="step2"):
            st.query_params["page"] = "step2"
            st.rerun()
    
    # Step 3: UID Matching
    with st.expander("**Step 3: 🎯 UID Matching**"):
        st.markdown("""
        **Purpose:** Match survey questions to question bank UIDs
        
        **What you'll do:**
        - Run TF-IDF and semantic matching algorithms
        - Review match confidence levels (High/Low/No match)
        - Handle identity questions automatically (16 identity types)
        - Resolve conflicts and duplicates
        
        **Prerequisites:** Complete Steps 1 & 2
        **Output:** Questions with assigned UIDs and match types
        """)
        
        if st.button("🎯 Start Step 3: UID Matching", key="step3"):
            st.query_params["page"] = "step3"
            st.rerun()
    
    # Step 4: Export Results
    with st.expander("**Step 4: 📤 Export Results**"):
        st.markdown("""
        **Purpose:** Export matched data with identity classification
        
        **What you'll do:**
        - Review final matched results
        - Download CSV/Excel exports
        - Separate identity vs non-identity questions
        - Generate comprehensive reports
        
        **Prerequisites:** Complete Steps 1, 2 & 3
        **Output:** Final export files ready for use
        """)
        
        if st.button("📊 Start Step 4: Export Results", key="step4"):
            st.query_params["page"] = "step4"
            st.rerun()
    
    st.markdown("---")
    
    # Recent Analytics
    try:
        analytics = get_snowflake_analytics()
        if analytics:
            st.markdown("### 📈 Quick Analytics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Questions", f"{analytics.get('total_questions', 0):,}")
            with col2:
                st.metric("Survey Stages", f"{analytics.get('survey_stages', 0)}")
            with col3:
                st.metric("Unique UIDs", f"{analytics.get('unique_uids', 0):,}")
            with col4:
                st.metric("Data Records", f"{analytics.get('total_records', 0):,}")
    except Exception as e:
        st.info("💡 Analytics will be available once Snowflake data is loaded")

def show_survey_selection():
    """Show SurveyMonkey survey selection page (like original uid_promax10.py)"""
    st.header("📋 Survey Selection & Question Bank")
    st.markdown("*Select SurveyMonkey surveys to extract questions from*")
    
    st.markdown('<div class="data-source-info">📊 <strong>Data Source:</strong> SurveyMonkey API - Survey selection and question extraction</div>', unsafe_allow_html=True)
    
    # Get SurveyMonkey token and check connection
    try:
        token = st.secrets["surveymonkey"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test connection
        response = requests.get("https://api.surveymonkey.com/v3/users/me", headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"❌ SurveyMonkey connection failed: {response.status_code}")
            return
        
        st.success("✅ SurveyMonkey connection established")
        
    except Exception as e:
        st.error(f"❌ SurveyMonkey connection failed: {e}")
        st.info("Please check your SurveyMonkey API token in secrets.toml")
        return
    
    # Get surveys with caching
    @st.cache_data(ttl=1800)
    def get_surveys_cached(token):
        """Get all surveys from SurveyMonkey API"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            all_surveys = []
            page = 1
            
            while True:
                params = {"per_page": 1000, "page": page}
                response = requests.get("https://api.surveymonkey.com/v3/surveys", 
                                      headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                page_surveys = data.get("data", [])
                
                if not page_surveys:
                    break
                    
                all_surveys.extend(page_surveys)
                total_surveys = data.get("total", 0)
                
                if len(all_surveys) >= total_surveys:
                    break
                    
                page += 1
                
            return all_surveys
            
        except Exception as e:
            logger.error(f"Failed to get surveys: {e}")
            return []
    
    # Load surveys
    with st.spinner("Loading SurveyMonkey surveys..."):
        surveys = get_surveys_cached(token)
    
    if not surveys:
        st.warning("⚠️ No surveys found. Check SurveyMonkey connection.")
        return
    
    st.success(f"✅ Found {len(surveys)} surveys in your SurveyMonkey account")
    
    # Survey Selection Interface
    st.subheader("🔍 Select Surveys to Extract Questions From")
    
    # Create survey options
    survey_options = [f"{s['id']} - {s['title']}" for s in surveys]
    selected_surveys = st.multiselect(
        "Choose surveys to analyze:",
        survey_options,
        help="Select surveys to extract questions from. Questions will be used for building question bank and UID matching."
    )
    
    # Extract selected survey IDs
    selected_survey_ids = [s.split(" - ")[0] for s in selected_surveys]
    
    if selected_survey_ids:
        st.info(f"📊 Selected {len(selected_survey_ids)} surveys for processing")
        
        # Process button
        if st.button("🚀 Extract Questions from Selected Surveys", type="primary"):
            extract_questions_from_surveys(selected_survey_ids, token)
    
    # Quick filters for survey selection
    with st.expander("🎛️ Survey Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by survey title
            title_filter = st.text_input("Filter by title:", key="survey_title_filter")
            if title_filter:
                filtered_surveys = [s for s in surveys if title_filter.lower() in s['title'].lower()]
                st.info(f"Found {len(filtered_surveys)} surveys matching '{title_filter}'")
        
        with col2:
            # Show survey count
            st.metric("Total Surveys", len(surveys))
            
            # Quick select options
            if st.button("Select All Template Surveys"):
                template_surveys = [f"{s['id']} - {s['title']}" for s in surveys if 'template' in s['title'].lower()]
                st.session_state.survey_multiselect = template_surveys
                st.rerun()

def extract_questions_from_surveys(survey_ids, token):
    """Extract questions from selected SurveyMonkey surveys"""
    
    @st.cache_data(ttl=600)
    def get_survey_details_with_retry(survey_id, token):
        """Get detailed survey information with retry logic"""
        try:
            url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get survey {survey_id}: {e}")
            return None
    
    def extract_questions_from_json(survey_json):
        """Extract questions from SurveyMonkey survey JSON"""
        questions = []
        
        for page in survey_json.get("pages", []):
            for question in page.get("questions", []):
                q_text = question.get("headings", [{}])[0].get("heading", "")
                q_id = question.get("id", None)
                family = question.get("family", None)
                
                # Determine schema type
                if family == "single_choice":
                    schema_type = "Single Choice"
                elif family == "multiple_choice":
                    schema_type = "Multiple Choice"
                elif family == "open_ended":
                    schema_type = "Open-Ended"
                elif family == "matrix":
                    schema_type = "Matrix"
                else:
                    choices = question.get("answers", {}).get("choices", [])
                    schema_type = "Multiple Choice" if choices else "Open-Ended"
                
                # Add main question
                questions.append({
                    "question_uid": q_id,
                    "question_text": q_text,
                    "schema_type": schema_type,
                    "is_choice": False,
                    "survey_id": survey_json.get("id"),
                    "survey_title": survey_json.get("title", "Unknown"),
                    "source": "surveymonkey"
                })
                
                # Add choice options
                for choice in question.get("answers", {}).get("choices", []):
                    choice_text = choice.get("text", "")
                    if choice_text.strip():
                        questions.append({
                            "question_uid": f"{q_id}_choice_{choice.get('id', '')}",
                            "question_text": f"{q_text} - {choice_text}",
                            "schema_type": schema_type,
                            "is_choice": True,
                            "survey_id": survey_json.get("id"),
                            "survey_title": survey_json.get("title", "Unknown"),
                            "source": "surveymonkey"
                        })
        
        return questions
    
    # Extract questions from selected surveys
    all_questions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, survey_id in enumerate(survey_ids):
        status_text.text(f"🔍 Processing survey {survey_id} ({i+1}/{len(survey_ids)})...")
        
        survey_json = get_survey_details_with_retry(survey_id, token)
        if survey_json:
            questions = extract_questions_from_json(survey_json)
            all_questions.extend(questions)
        
        progress_bar.progress((i + 1) / len(survey_ids))
    
    progress_bar.empty()
    status_text.empty()
    
    if all_questions:
        # Convert to DataFrame and store in session
        questions_df = pd.DataFrame(all_questions)
        st.session_state.surveymonkey_questions = questions_df
        
        # Success metrics
        st.success(f"✅ Extracted {len(all_questions)} questions from {len(survey_ids)} surveys!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            main_questions = len(questions_df[questions_df["is_choice"] == False])
            st.metric("❓ Main Questions", main_questions)
        with col2:
            choices = len(questions_df[questions_df["is_choice"] == True])
            st.metric("🔘 Choice Options", choices)
        with col3:
            unique_surveys = questions_df["survey_id"].nunique()
            st.metric("📊 Surveys", unique_surveys)
        
        # Preview
        st.subheader("🔍 Questions Preview")
        show_main_only = st.checkbox("Show main questions only", value=True)
        display_df = questions_df[questions_df["is_choice"] == False] if show_main_only else questions_df
        
        st.dataframe(
            display_df[["question_text", "schema_type", "survey_title"]].head(20),
            use_container_width=True
        )
        
        # Next steps
        st.subheader("➡️ Next Steps")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏗️ Build Question Bank", type="primary"):
                # Use SurveyMonkey questions for question bank
                st.session_state.current_page = 'question_bank_builder'
                st.rerun()
        
        with col2:
            if st.button("🎯 Start UID Matching"):
                st.session_state.current_page = 'uid_matching'
                st.rerun()
        
    else:
        st.error("❌ No questions extracted from selected surveys")

if __name__ == "__main__":
    main() 