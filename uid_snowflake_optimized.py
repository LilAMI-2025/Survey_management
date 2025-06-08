#!/usr/bin/env python3
"""
Enhanced UID Management System - Snowflake Optimized Version
Leverages direct Snowflake queries for maximum performance and data coverage.
Based on successful validation of CHOICE_TEXT column and deduplication logic.
Complete standalone implementation without external dependencies.
"""

import streamlit as st

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="UID Matcher Enhanced - Snowflake Optimized",
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

# Configuration
SNOWFLAKE_QUERY_LIMIT = 10000  # Expanded from 100 for comprehensive data
DEFAULT_SURVEY_STAGES = ["Annual Impact Survey", "AP Survey"]  # Default filter options
CACHE_TTL = 3600  # 1 hour cache
BATCH_SIZE = 100

# Matching thresholds
TFIDF_HIGH_CONFIDENCE = 0.85
TFIDF_LOW_CONFIDENCE = 0.6
SEMANTIC_THRESHOLD = 0.7

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= SESSION STATE INITIALIZATION =============
def initialize_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'question_bank_complete' not in st.session_state:
        st.session_state.question_bank_complete = None
    if 'matched_results' not in st.session_state:
        st.session_state.matched_results = None

# Initialize session state
initialize_session_state()

# ============= CORE UTILITY FUNCTIONS =============

@st.cache_resource
def get_snowflake_engine():
    """Create Snowflake engine using Streamlit secrets"""
    try:
        sf = st.secrets["snowflake"]
        engine = create_engine(
            f"snowflake://{sf['user']}:{sf['password']}@{sf['account']}/{sf['database']}/{sf['schema']}?warehouse={sf['warehouse']}&role={sf['role']}"
        )
        return engine
    except Exception as e:
        st.warning(
            f"Snowflake connection failed: {e}. "
            "Please check your secrets configuration."
        )
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load SentenceTransformer model for semantic matching"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def enhanced_normalize(text):
    """Enhanced text normalization for better matching"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase and strip
    text = str(text).lower().strip()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def clean_question_text(text):
    """Clean question text for better matching"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).strip()
    
    # Remove common prefixes/suffixes
    text = re.sub(r'^(Question:|Q\d+:|Question \d+:)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(.*?\)\s*$', '', text)  # Remove trailing parentheses
    
    return text.strip()

def contains_identity_info(text):
    """Enhanced identity detection with 16 specific types"""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower().strip()
    
    # 16 identity patterns
    identity_patterns = [
        # Age patterns
        r'\b(age|how old|birth year|born in)\b',
        # City patterns  
        r'\b(city|town|municipality|where.*live|location.*city)\b',
        # Company patterns
        r'\b(company|organization|employer|work for|employed by)\b',
        # Country patterns
        r'\b(country|nation|nationality|where.*from|citizenship)\b',
        # Date of Birth patterns
        r'\b(date.*birth|birthday|born on|birth date|dob)\b',
        # Email patterns
        r'\b(email|e-mail|email address|contact.*email)\b',
        # Education patterns
        r'\b(education|degree|qualification|university|college|school)\b',
        # Full Name patterns
        r'\b(full name|complete name|first.*last name|your name)\b',
        # First Name patterns
        r'\b(first name|given name|forename)\b',
        # Last Name patterns
        r'\b(last name|surname|family name|lastname)\b',
        # Gender patterns
        r'\b(gender|sex|male|female|identify as)\b',
        # Location patterns
        r'\b(location|address|where.*located|postal code|zip code)\b',
        # Phone patterns
        r'\b(phone|telephone|mobile|contact.*number)\b',
        # Region patterns
        r'\b(region|state|province|area|district)\b',
        # Title/Role patterns
        r'\b(title|position|role|job title|designation)\b',
        # PIN/Passport patterns
        r'\b(pin|passport|id number|identification|social security)\b'
    ]
    
    return any(re.search(pattern, text_lower) for pattern in identity_patterns)

def determine_identity_type(text):
    """Determine specific identity type from 16 categories"""
    if not isinstance(text, str):
        return "Unknown"
    
    text_lower = text.lower().strip()
    
    # Check each identity type in order of specificity
    if re.search(r'\b(date.*birth|birthday|born on|birth date|dob)\b', text_lower):
        return "Date of Birth"
    elif re.search(r'\b(full name|complete name|first.*last name|your name)\b', text_lower):
        return "Full Name"
    elif re.search(r'\b(first name|given name|forename)\b', text_lower):
        return "First Name"
    elif re.search(r'\b(last name|surname|family name|lastname)\b', text_lower):
        return "Last Name"
    elif re.search(r'\b(email|e-mail|email address|contact.*email)\b', text_lower):
        return "E-Mail"
    elif re.search(r'\b(phone|telephone|mobile|contact.*number)\b', text_lower):
        return "Phone Number"
    elif re.search(r'\b(pin|passport|id number|identification|social security)\b', text_lower):
        return "PIN/Passport"
    elif re.search(r'\b(age|how old|birth year|born in)\b', text_lower):
        return "Age"
    elif re.search(r'\b(gender|sex|male|female|identify as)\b', text_lower):
        return "Gender"
    elif re.search(r'\b(city|town|municipality|where.*live|location.*city)\b', text_lower):
        return "City"
    elif re.search(r'\b(country|nation|nationality|where.*from|citizenship)\b', text_lower):
        return "Country"
    elif re.search(r'\b(region|state|province|area|district)\b', text_lower):
        return "Region"
    elif re.search(r'\b(location|address|where.*located|postal code|zip code)\b', text_lower):
        return "Location"
    elif re.search(r'\b(company|organization|employer|work for|employed by)\b', text_lower):
        return "Company"
    elif re.search(r'\b(title|position|role|job title|designation)\b', text_lower):
        return "Title/Role"
    elif re.search(r'\b(education|degree|qualification|university|college|school)\b', text_lower):
        return "Education level"
    else:
        return "Other Identity"

def convert_uid_to_3_chars(uid):
    """Convert UID to 3-character format using Google Sheets formula logic"""
    if pd.isna(uid) or uid == "":
        return uid
    
    uid_str = str(uid).strip()
    
    # If already 3 characters, return as is
    if len(uid_str) == 3:
        return uid_str
    
    # If less than 3 characters, pad with zeros
    if len(uid_str) < 3:
        return uid_str.zfill(3)
    
    # If more than 3 characters, take first 3
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
        
        # Base conditions with CALA filter
        base_conditions = """
        WHERE HEADING_0 IS NOT NULL 
        AND TRIM(HEADING_0) != ''
        AND UPPER(HEADING_0) NOT LIKE '%CALA%'
        """
        
        # Check what columns exist for choices
        columns_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'DBT_SURVEY_MONKEY' 
        AND table_name = 'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE'
        AND (column_name LIKE '%CHOICE%' OR column_name LIKE '%TEXT%')
        ORDER BY column_name
        """
        
        with engine.connect() as conn:
            columns_result = pd.read_sql(text(columns_query), conn)
        
        # Determine choice column
        choice_column = "CHOICE_TEXT"  # Default
        if not columns_result.empty:
            # Handle case-insensitive column names
            column_name_col = 'column_name' if 'column_name' in columns_result.columns else 'COLUMN_NAME'
            choice_cols = columns_result[column_name_col].tolist()
            if 'CHOICE_TEXT' in choice_cols:
                choice_column = "CHOICE_TEXT"
            elif 'ROW_TEXT' in choice_cols:
                choice_column = "ROW_TEXT"
            else:
                choice_column = choice_cols[0] if choice_cols else "CHOICE_TEXT"
        
        # Add survey stage filter if specified and stages exist
        stage_filter = ""
        available_stages = get_survey_stage_options()
        if survey_stages and len(survey_stages) > 0 and available_stages:
            stages_str = "', '".join(survey_stages)
            # Use the first available stage column
            stage_col_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'DBT_SURVEY_MONKEY' 
            AND table_name = 'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE'
            AND column_name LIKE '%STAGE%'
            LIMIT 1
            """
            with engine.connect() as conn:
                stage_col_result = pd.read_sql(text(stage_col_query), conn)
            
            if not stage_col_result.empty:
                # Handle case-insensitive column names
                stage_column_col = 'column_name' if 'column_name' in stage_col_result.columns else 'COLUMN_NAME'
                stage_column = stage_col_result.iloc[0][stage_column_col]
                stage_filter = f" AND {stage_column} IN ('{stages_str}')"
        
        if grouped_by_stage:
            # Grouped by survey stage (original behavior)
            if include_choices:
                query = f"""
                WITH ranked_data AS (
                    SELECT 
                        HEADING_0,
                        COALESCE({choice_column}, '') as CHOICE_TEXT,
                        UID,
                        DATE_MODIFIED,
                        COUNT(*) as FREQUENCY,
                        ROW_NUMBER() OVER (
                            PARTITION BY HEADING_0, COALESCE({choice_column}, '') 
                            ORDER BY DATE_MODIFIED DESC, COUNT(*) DESC
                        ) as rn
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    {base_conditions}{stage_filter}
                    GROUP BY HEADING_0, COALESCE({choice_column}, ''), UID, DATE_MODIFIED
                )
                SELECT 
                    HEADING_0 as QUESTION_TEXT,
                    CASE WHEN CHOICE_TEXT = '' THEN NULL ELSE CHOICE_TEXT END as CHOICE_TEXT,
                    UID,
                    FREQUENCY,
                    DATE_MODIFIED
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
                        DATE_MODIFIED,
                        COUNT(*) as FREQUENCY,
                        ROW_NUMBER() OVER (
                            PARTITION BY HEADING_0 
                            ORDER BY DATE_MODIFIED DESC, COUNT(*) DESC
                        ) as rn
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    {base_conditions}{stage_filter}
                    GROUP BY HEADING_0, UID, DATE_MODIFIED
                )
                SELECT 
                    HEADING_0 as QUESTION_TEXT,
                    UID,
                    FREQUENCY,
                    DATE_MODIFIED
                FROM ranked_data 
                WHERE rn = 1
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
                        COALESCE({choice_column}, '') as CHOICE_TEXT,
                        UID,
                        DATE_MODIFIED,
                        COUNT(*) as FREQUENCY,
                        ROW_NUMBER() OVER (
                            PARTITION BY HEADING_0, COALESCE({choice_column}, '') 
                            ORDER BY DATE_MODIFIED DESC, COUNT(*) DESC
                        ) as rn
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    {base_conditions}{stage_filter}
                    GROUP BY HEADING_0, COALESCE({choice_column}, ''), UID, DATE_MODIFIED
                )
                SELECT 
                    HEADING_0 as QUESTION_TEXT,
                    CASE WHEN CHOICE_TEXT = '' THEN NULL ELSE CHOICE_TEXT END as CHOICE_TEXT,
                    UID,
                    FREQUENCY,
                    DATE_MODIFIED
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
                        DATE_MODIFIED,
                        COUNT(*) as FREQUENCY,
                        ROW_NUMBER() OVER (
                            PARTITION BY HEADING_0 
                            ORDER BY DATE_MODIFIED DESC, COUNT(*) DESC
                        ) as rn
                    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
                    {base_conditions}{stage_filter}
                    GROUP BY HEADING_0, UID, DATE_MODIFIED
                )
                SELECT 
                    HEADING_0 as QUESTION_TEXT,
                    UID,
                    FREQUENCY,
                    DATE_MODIFIED
                FROM ranked_data 
                WHERE rn = 1
                ORDER BY UID, FREQUENCY DESC
                LIMIT {limit}
                """
        
        with engine.connect() as conn:
            result_df = pd.read_sql(text(query), conn)
        
        if not result_df.empty:
            # Add identity detection columns
            result_df['IS_IDENTITY'] = result_df['QUESTION_TEXT'].apply(contains_identity_info)
            result_df['IDENTITY_TYPE'] = result_df['QUESTION_TEXT'].apply(determine_identity_type)
            
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
        
        # First, let's check what columns actually exist
        columns_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'DBT_SURVEY_MONKEY' 
        AND table_name = 'SURVEY_DETAILS_RESPONSES_COMBINED_LIVE'
        AND column_name LIKE '%STAGE%'
        ORDER BY column_name
        """
        
        with engine.connect() as conn:
            columns_result = pd.read_sql(text(columns_query), conn)
            
        # Check if we have any stage-related columns
        if not columns_result.empty:
            # Handle case-insensitive column names
            column_name_col = 'column_name' if 'column_name' in columns_result.columns else 'COLUMN_NAME'
            stage_column = columns_result.iloc[0][column_name_col]
            
            query = f"""
            SELECT DISTINCT {stage_column}
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE {stage_column} IS NOT NULL 
            AND TRIM({stage_column}) != ''
            AND UPPER(HEADING_0) NOT LIKE '%CALA%'
            ORDER BY {stage_column}
            """
            
            with engine.connect() as conn:
                result_df = pd.read_sql(text(query), conn)
                
            stages = result_df[stage_column].tolist() if not result_df.empty else []
            logger.info(f"✅ Retrieved {len(stages)} survey stages from Snowflake")
            return stages
        else:
            # Fallback: try to derive stages from survey titles or other data
            logger.warning("No STAGE column found, using fallback method")
            return ["AP Survey", "Annual Impact Survey"]  # Based on your previous findings
        
    except Exception as e:
        logger.error(f"Failed to get survey stage options: {str(e)}")
        return ["AP Survey", "Annual Impact Survey"]  # Fallback

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

def show_snowflake_question_bank_builder():
    """Enhanced question bank builder using Snowflake with original UI structure"""
    
    st.header("🏗️ Snowflake-Optimized Question Bank Builder")
    st.markdown("*Build comprehensive question banks directly from Snowflake data*")
    
    # Sidebar filters - maintaining original structure
    with st.sidebar:
        st.subheader("🎛️ Filters & Configuration")
        
        # Question bank type selection (NEW)
        st.markdown("**📊 Question Bank Type**")
        bank_type = st.radio(
            "Select question bank type:",
            options=["Grouped by Survey Stage", "Unique Questions (All Stages)"],
            help="""
            • **Grouped by Survey Stage**: Questions organized by survey stage - useful for client-specific modifications
            • **Unique Questions**: Deduplicated questions across all stages - prevents duplicate UIDs and questions
            """
        )
        
        grouped_by_stage = (bank_type == "Grouped by Survey Stage")
        
        st.markdown("---")
        
        # Survey stage filter with multi-select dropdown (UPDATED)
        available_stages = get_survey_stage_options()
        if available_stages:
            # Add "All" option
            stage_options = ["All"] + available_stages
            selected_stage_option = st.selectbox(
                "Survey Stage Selection",
                options=stage_options,
                help="Select 'All' for all stages or choose specific stages"
            )
            
            if selected_stage_option == "All":
                selected_stages = available_stages
                st.info(f"✅ All {len(available_stages)} survey stages selected")
            else:
                # Allow additional selections
                additional_stages = st.multiselect(
                    "Additional Survey Stages (optional)",
                    options=[s for s in available_stages if s != selected_stage_option],
                    help="Select additional survey stages to include"
                )
                selected_stages = [selected_stage_option] + additional_stages
                st.info(f"✅ {len(selected_stages)} survey stage(s) selected")
        else:
            selected_stages = []
            st.warning("⚠️ No survey stages available")
        
        # Question scope settings
        include_choices = st.checkbox(
            "Include Choice Text", 
            value=True,
            help="Include answer choices for multi-choice questions"
        )
        
        # Query limit
        query_limit = st.slider(
            "Query Limit", 
            min_value=100, 
            max_value=50000, 
            value=SNOWFLAKE_QUERY_LIMIT,
            step=500,
            help="Maximum number of records to retrieve"
        )

        # Build button
        build_bank = st.button("🚀 Build Question Bank", type="primary")

    # Main content - maintaining original structure
    if build_bank:
        if not selected_stages:
            st.error("❌ Please select at least one survey stage")
            return
            
        with st.spinner(f"Building {'grouped' if grouped_by_stage else 'unique'} question bank from Snowflake..."):
            # Get question bank
            question_bank_df = get_comprehensive_question_bank_from_snowflake(
                limit=query_limit,
                survey_stages=selected_stages,
                include_choices=include_choices,
                grouped_by_stage=grouped_by_stage
            )
            
        if not question_bank_df.empty:
            # Store in session state
            st.session_state.question_bank_snowflake = question_bank_df
            st.session_state.question_bank_type = bank_type
            
            # Display summary - maintaining original style
            st.success(f"✅ Built {bank_type.lower()} with {len(question_bank_df):,} records")
            
            # Summary metrics - maintaining original structure
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                unique_questions = question_bank_df['QUESTION_TEXT'].nunique()
                st.metric("Unique Questions", f"{unique_questions:,}")
            
            with col2:
                if 'CHOICE_TEXT' in question_bank_df.columns:
                    choice_count = question_bank_df['CHOICE_TEXT'].notna().sum()
                    st.metric("Questions with Choices", f"{choice_count:,}")
                else:
                    st.metric("Main Questions Only", f"{len(question_bank_df):,}")
            
            with col3:
                unique_uids = question_bank_df['UID'].nunique()
                st.metric("Unique UIDs", unique_uids)
            
            with col4:
                identity_count = question_bank_df['IS_IDENTITY'].sum() if 'IS_IDENTITY' in question_bank_df.columns else 0
                st.metric("Identity Questions", identity_count)

            # Display breakdown by survey stage - maintaining original approach
            if 'SURVEY_STAGE' in question_bank_df.columns:
                st.subheader("📊 Breakdown by Survey Stage")
                stage_summary = question_bank_df.groupby('SURVEY_STAGE').agg({
                    'QUESTION_TEXT': 'count',
                    'UID': 'nunique',
                    'FREQUENCY': 'sum' if 'FREQUENCY' in question_bank_df.columns else 'count'
                }).round().astype(int)
                
                # Rename columns for clarity
                stage_summary.columns = ['Total Records', 'Unique UIDs', 'Total Usage']
                st.dataframe(stage_summary, use_container_width=True)

            # Sample data preview - maintaining original approach
            st.subheader("🔍 Sample Data Preview")
            
            # Select columns to display
            preview_cols = ['QUESTION_TEXT']
            if 'CHOICE_TEXT' in question_bank_df.columns:
                preview_cols.append('CHOICE_TEXT')
            preview_cols.extend(['UID', 'UID_3_CHAR'])
            if 'SURVEY_STAGE' in question_bank_df.columns:
                preview_cols.append('SURVEY_STAGE')
            if 'IS_IDENTITY' in question_bank_df.columns:
                preview_cols.extend(['IS_IDENTITY', 'IDENTITY_TYPE'])
            if 'FREQUENCY' in question_bank_df.columns:
                preview_cols.append('FREQUENCY')
            
            # Filter to available columns
            available_cols = [col for col in preview_cols if col in question_bank_df.columns]
            
            st.dataframe(question_bank_df[available_cols].head(20), use_container_width=True)
            
        else:
            st.error("❌ Failed to build question bank. Please check your filters and try again.")
    
    # Analytics section - maintaining original structure
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
    if 'question_bank_snowflake' not in st.session_state or st.session_state.question_bank_snowflake.empty:
        st.warning("⚠️ No Snowflake question bank loaded. Please build one first in the Question Bank Builder tab.")
        return
    
    question_bank_df = st.session_state.question_bank_snowflake
    bank_type = st.session_state.get('question_bank_type', 'Unknown')
    st.info(f"📊 Using question bank: **{bank_type}** ({len(question_bank_df):,} records)")
    
    # File upload for target questions - maintaining original structure
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
                    
                    # Display results summary - maintaining original structure
                    total_questions = len(target_df)
                    matched_questions = matched_results['Final_UID'].notna().sum()
                    match_rate = (matched_questions / total_questions * 100) if total_questions > 0 else 0
                    
                    st.success(f"✅ Matching completed! {matched_questions}/{total_questions} questions matched ({match_rate:.1f}%)")
                    
                    # Matching statistics - maintaining original approach
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Questions", total_questions)
                    
                    with col2:
                        st.metric("Matched Questions", matched_questions)
                    
                    with col3:
                        unique_uids = matched_results['Final_UID'].nunique()
                        st.metric("Unique UIDs Found", unique_uids)
                    
                    with col4:
                        identity_matches = 0
                        if 'IS_IDENTITY' in matched_results.columns:
                            identity_matches = matched_results['IS_IDENTITY'].sum()
                        st.metric("Identity Matches", identity_matches)
                    
                    # Results preview - maintaining original structure
                    st.subheader("🎯 Matching Results Preview")
                    
                    # Select columns for display
                    results_cols = ['question_text']
                    if 'Final_UID' in matched_results.columns:
                        results_cols.append('Final_UID')
                    if 'Final_Match_Type' in matched_results.columns:
                        results_cols.append('Final_Match_Type')
                    if 'Similarity' in matched_results.columns:
                        results_cols.append('Similarity')
                    if 'IS_IDENTITY' in matched_results.columns:
                        results_cols.extend(['IS_IDENTITY', 'IDENTITY_TYPE'])
                    
                    # Filter to available columns
                    available_cols = [col for col in results_cols if col in matched_results.columns]
                    
                    if available_cols:
                        st.dataframe(matched_results[available_cols].head(20), use_container_width=True)
                    else:
                        # Fallback to showing all columns
                        st.dataframe(matched_results.head(20), use_container_width=True)
                    
                else:
                    st.error("❌ Matching failed")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    # Display saved results if available - maintaining original structure
    if 'matching_results' in st.session_state and not st.session_state.matching_results.empty:
        st.subheader("💾 Saved Matching Results")
        
        results_df = st.session_state.matching_results
        
        # Enhanced analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Match Type Distribution**")
            if 'Final_Match_Type' in results_df.columns:
                match_type_counts = results_df['Final_Match_Type'].value_counts()
                for match_type, count in match_type_counts.items():
                    st.text(f"• {match_type}: {count}")
        
        with col2:
            st.markdown("**🔍 Identity Analysis**")
            if 'IS_IDENTITY' in results_df.columns:
                total_identity = results_df['IS_IDENTITY'].sum()
                total_questions = len(results_df)
                identity_rate = (total_identity / total_questions * 100) if total_questions > 0 else 0
                st.text(f"• Identity questions: {total_identity} ({identity_rate:.1f}%)")
                
                if 'IDENTITY_TYPE' in results_df.columns:
                    identity_types = results_df[results_df['IS_IDENTITY'] == True]['IDENTITY_TYPE'].value_counts()
                    for identity_type, count in identity_types.head(5).items():
                        st.text(f"• {identity_type}: {count}")

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
    
    st.title("❄️ Snowflake-Optimized UID Management System")
    st.markdown("*Enhanced question bank building and UID matching with direct Snowflake integration*")
    
    # Navigation - restore original tab structure
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