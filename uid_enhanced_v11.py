# ============= ENHANCED UID MATCHER V11 =============
# Hybrid Approach: SurveyMonkey Question Bank + Snowflake UID References
# Enhanced with comprehensive question structure and improved matching

import streamlit as st
import pandas as pd
import requests
import re
import logging
import json
import time
import os
import numpy as np
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections import defaultdict, Counter
from datetime import datetime, date
from difflib import SequenceMatcher

# ============= STREAMLIT CONFIGURATION =============
st.set_page_config(
    page_title="Enhanced UID Matcher V11", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# ============= LOGGING SETUP =============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= CONSTANTS AND CONFIGURATION =============

# Enhanced matching thresholds
TFIDF_HIGH_CONFIDENCE = 0.65
TFIDF_LOW_CONFIDENCE = 0.55
SEMANTIC_THRESHOLD = 0.65
QUESTION_QUALITY_THRESHOLD = 60
BATCH_SIZE = 500

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_FILE = "enhanced_survey_cache_v11.json"
REQUEST_DELAY = 0.3

# Identity types (21 categories as specified)
IDENTITY_TYPES = [
    'Full Name', 'First Name', 'Last Name', 'E-Mail', 'Company', 'Gender', 
    'Country', 'Age', 'Title/Role', 'Phone Number', 'Location', 
    'PIN/Passport', 'Date of Birth', 'UCT Student Number',
    'Department', 'Region', 'City', 'ID Number', 'Marital Status',
    'Education level', 'English Proficiency'
]

# Enhanced UID Final Reference
UID_FINAL_REFERENCE = {
    "On a scale of 0-10, how likely is it that you would recommend AMI to someone (a colleague, friend or other business?)": 1,
    "Do you (in general) feel more confident about your ability to raise capital for your business?": 38,
    "Have you set and shared your Growth Goal with AMI?": 57,
    "Have you observed an improvement in the following areas in your business since engaging with AMI?": 77,
    "What is your gender?": 233,
    "What is your age?": 234,
    "What is your job title/role in the business?": 335,
    "Your email address:": 350,
    "Please select your country area code followed by your cellphone number with no spaces (eg. 794000000):": 360,
    # Add more as needed
}

# ============= CSS STYLES =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .enhanced-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .hybrid-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .question-bank-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= SESSION STATE INITIALIZATION =============
def initialize_session_state():
    """Initialize enhanced session state"""
    defaults = {
        "page": "enhanced_home",
        "question_bank_complete": None,
        "surveymonkey_questions": None,
        "snowflake_references": None,
        "matched_results": None,
        "hybrid_mode": True,
        "enhanced_cache": {},
        "selected_surveys_v11": [],
        "comprehensive_question_db": None,
        "uid_final_reference": UID_FINAL_REFERENCE
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============= CACHED RESOURCES =============
@st.cache_resource
def load_enhanced_model():
    """Load enhanced sentence transformer model"""
    logger.info(f"Loading Enhanced SentenceTransformer: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_enhanced_snowflake_engine():
    """Enhanced Snowflake connection"""
    try:
        sf = st.secrets["snowflake"]
        engine = create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
            f"?warehouse={sf.warehouse}&role={sf.role}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT CURRENT_VERSION()"))
        logger.info("Enhanced Snowflake connection established")
        return engine
    except Exception as e:
        logger.error(f"Enhanced Snowflake connection failed: {e}")
        raise

# ============= ENHANCED UTILITY FUNCTIONS =============
def enhanced_normalize_text(text, synonym_map=None):
    """Enhanced text normalization with context awareness"""
    if not isinstance(text, str):
        return ""
    
    # Clean and normalize
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Apply synonym mapping if provided
    if synonym_map:
        for phrase, replacement in synonym_map.items():
            text = text.replace(phrase, replacement)
    
    # Remove stop words but keep important question words
    important_words = {'what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'are', 'is'}
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS or w in important_words]
    
    return ' '.join(words)

def enhanced_quality_score(question):
    """Enhanced question quality scoring with multiple criteria"""
    if not isinstance(question, str) or len(question.strip()) < 3:
        return 0
    
    score = 0
    text = question.lower().strip()
    
    # Length scoring (sweet spot 10-80 characters)
    length = len(question)
    if 10 <= length <= 80:
        score += 30
    elif 5 <= length <= 120:
        score += 20
    
    # Question structure
    if question.strip().endswith('?'):
        score += 25
    
    # Question words at start
    question_starters = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'are', 'is', 'can', 'will', 'would']
    first_word = text.split()[0] if text.split() else ""
    if first_word in question_starters:
        score += 20
    
    # Proper capitalization
    if question and question[0].isupper():
        score += 10
    
    # Complete sentence structure
    word_count = len(question.split())
    if 3 <= word_count <= 12:
        score += 15
    
    # Avoid artifacts
    bad_patterns = ['click here', 'please select', '...', 'html', '<div', '<span']
    if any(pattern in text for pattern in bad_patterns):
        score -= 30
    
    return max(0, score)

def convert_uid_to_3_chars(uid):
    """Convert UID to 3 chars using Google Sheets formula"""
    if pd.isna(uid) or uid is None:
        return uid
    
    uid_str = str(uid).strip()
    
    if len(uid_str) <= 3:
        return uid_str
    elif len(uid_str) == 4:
        return uid_str[:1]
    elif len(uid_str) == 5:
        return uid_str[:2]
    elif len(uid_str) > 5:
        return uid_str[:3]
    else:
        return uid_str

# ============= IDENTITY DETECTION FUNCTIONS =============
def detect_identity_question(text):
    """Enhanced identity question detection"""
    if not isinstance(text, str):
        return False, None
    
    text_lower = text.lower().strip()
    
    # Identity patterns with their types
    identity_patterns = {
        'First Name': [r'first\s+name', r'what\s+is\s+your\s+first\s+name'],
        'Last Name': [r'last\s+name', r'surname', r'what\s+is\s+your\s+last\s+name'],
        'Full Name': [r'full\s+name', r'what\s+is\s+your\s+name'],
        'E-Mail': [r'email', r'e-mail', r'mail\s+address'],
        'Phone Number': [r'phone', r'mobile', r'telephone'],
        'Age': [r'what\s+is\s+your\s+age', r'how\s+old\s+are\s+you'],
        'Gender': [r'gender', r'sex', r'male\s+or\s+female'],
        'Company': [r'company', r'organization', r'organisation'],
        'Country': [r'country', r'which\s+country'],
        'City': [r'city', r'which\s+city'],
        'Region': [r'region', r'which\s+region'],
        'Department': [r'department', r'which\s+department'],
        'Title/Role': [r'job\s+title', r'role', r'position'],
        'Location': [r'location', r'address'],
        'ID Number': [r'id\s+number', r'identification'],
        'Date of Birth': [r'date\s+of\s+birth', r'dob', r'when\s+were\s+you\s+born'],
        'Marital Status': [r'marital\s+status', r'married'],
        'Education level': [r'education\s+level', r'qualification'],
        'English Proficiency': [r'english\s+proficiency', r'language\s+proficiency']
    }
    
    for identity_type, patterns in identity_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, identity_type
    
    return False, None

# ============= SURVEYMONKEY ENHANCED FUNCTIONS =============
def get_enhanced_surveymonkey_token():
    """Enhanced SurveyMonkey token retrieval"""
    try:
        return st.secrets["surveymonkey"]["access_token"]
    except Exception as e:
        logger.error(f"Failed to get SurveyMonkey token: {e}")
        return None

@st.cache_data(ttl=1800)
def get_enhanced_surveys_from_2023():
    """Get surveys from 2023 onwards with enhanced filtering"""
    token = get_enhanced_surveymonkey_token()
    if not token:
        return []
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get surveys with date filtering
        params = {
            "per_page": 1000,  # Get more surveys at once
            "sort_order": "DESC",
            "sort_by": "date_created"
        }
        
        response = requests.get("https://api.surveymonkey.com/v3/surveys", 
                              headers=headers, params=params)
        response.raise_for_status()
        all_surveys = response.json().get("data", [])
        
        # Filter surveys from 2023 onwards
        filtered_surveys = []
        cutoff_date = datetime(2023, 1, 1)
        
        for survey in all_surveys:
            date_created = survey.get("date_created")
            if date_created:
                try:
                    # Parse SurveyMonkey date format
                    survey_date = datetime.strptime(date_created, "%Y-%m-%dT%H:%M:%S")
                    if survey_date >= cutoff_date:
                        filtered_surveys.append(survey)
                except ValueError:
                    # If date parsing fails, include the survey to be safe
                    filtered_surveys.append(survey)
        
        logger.info(f"Retrieved {len(filtered_surveys)} surveys from 2023+ (out of {len(all_surveys)} total)")
        return filtered_surveys
        
    except Exception as e:
        logger.error(f"Failed to get surveys from 2023+: {e}")
        return []

# Backwards compatibility
@st.cache_data(ttl=1800)
def get_enhanced_surveys():
    """Backwards compatibility - calls get_enhanced_surveys_from_2023"""
    return get_enhanced_surveys_from_2023()

def compute_question_similarity(text1, text2):
    """Compute similarity between two questions using difflib"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    norm_text1 = enhanced_normalize_text(text1).lower()
    norm_text2 = enhanced_normalize_text(text2).lower()
    
    # Use SequenceMatcher for similarity (built-in Python)
    similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    return similarity

@st.cache_data(ttl=3600)
def compute_semantic_similarities(questions_list):
    """Compute semantic similarities using sentence transformers"""
    if not questions_list:
        return np.array([])
    
    model = load_enhanced_model()
    embeddings = model.encode(questions_list, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    return similarity_matrix

def deduplicate_questions_advanced(all_questions_df):
    """Advanced deduplication using sentence transformers and fuzzy matching"""
    if all_questions_df.empty:
        return all_questions_df
    
    st.info("🔄 Performing advanced deduplication using AI and fuzzy matching...")
    
    # Separate main questions and choices for different deduplication strategies
    main_questions = all_questions_df[all_questions_df["is_choice"] == False].copy()
    choices = all_questions_df[all_questions_df["is_choice"] == True].copy()
    
    # Deduplicate main questions
    dedup_main = deduplicate_main_questions(main_questions)
    
    # Deduplicate choices
    dedup_choices = deduplicate_choices(choices)
    
    # Combine results
    result_df = pd.concat([dedup_main, dedup_choices], ignore_index=True)
    
    logger.info(f"📊 Deduplication results: {len(all_questions_df)} → {len(result_df)} questions")
    return result_df

def deduplicate_main_questions(main_questions_df):
    """Deduplicate main questions using semantic similarity"""
    if main_questions_df.empty:
        return main_questions_df
    
    questions_list = main_questions_df["question_text"].tolist()
    
    # Compute semantic similarities
    semantic_matrix = compute_semantic_similarities(questions_list)
    
    # Find duplicates
    duplicates_to_remove = set()
    SEMANTIC_SIMILARITY_THRESHOLD = 0.85  # High threshold for semantic similarity
    TEXT_SIMILARITY_THRESHOLD = 0.80  # Adjusted threshold for text similarity
    
    for i in range(len(questions_list)):
        if i in duplicates_to_remove:
            continue
            
        for j in range(i + 1, len(questions_list)):
            if j in duplicates_to_remove:
                continue
            
            # Check semantic similarity
            semantic_sim = semantic_matrix[i][j] if semantic_matrix.size > 0 else 0
            
            # Check text similarity using difflib
            text_sim = compute_question_similarity(questions_list[i], questions_list[j])
            
            # Combined decision - either high semantic OR high text similarity
            if semantic_sim > SEMANTIC_SIMILARITY_THRESHOLD or text_sim > TEXT_SIMILARITY_THRESHOLD:
                # Keep the higher quality question
                if main_questions_df.iloc[i]["quality_score"] >= main_questions_df.iloc[j]["quality_score"]:
                    duplicates_to_remove.add(j)
                else:
                    duplicates_to_remove.add(i)
                    break
    
    # Remove duplicates
    indices_to_keep = [i for i in range(len(main_questions_df)) if i not in duplicates_to_remove]
    result_df = main_questions_df.iloc[indices_to_keep].reset_index(drop=True)
    
    st.success(f"✅ Main questions: {len(main_questions_df)} → {len(result_df)} (removed {len(duplicates_to_remove)} duplicates)")
    return result_df

def deduplicate_choices(choices_df):
    """Deduplicate choices with context awareness"""
    if choices_df.empty:
        return choices_df
    
    # Group choices by normalized choice text
    choice_groups = {}
    CHOICE_SIMILARITY_THRESHOLD = 0.85  # Adjusted threshold for choices using difflib
    
    for idx, row in choices_df.iterrows():
        choice_text = row["choice_text"]
        normalized_choice = enhanced_normalize_text(choice_text).lower()
        
        # Find similar existing choices
        matched_group = None
        for existing_choice in choice_groups.keys():
            similarity = compute_question_similarity(normalized_choice, existing_choice)
            if similarity > CHOICE_SIMILARITY_THRESHOLD:
                matched_group = existing_choice
                break
        
        if matched_group:
            choice_groups[matched_group].append(idx)
        else:
            choice_groups[normalized_choice] = [idx]
    
    # Keep best representative from each group
    indices_to_keep = []
    for group_indices in choice_groups.values():
        if len(group_indices) == 1:
            indices_to_keep.append(group_indices[0])
        else:
            # Keep the choice with highest quality score
            best_idx = max(group_indices, 
                          key=lambda x: choices_df.iloc[x]["quality_score"])
            indices_to_keep.append(best_idx)
    
    result_df = choices_df.iloc[indices_to_keep].reset_index(drop=True)
    
    removed_count = len(choices_df) - len(result_df)
    st.success(f"✅ Choices: {len(choices_df)} → {len(result_df)} (removed {removed_count} duplicates)")
    return result_df

def extract_enhanced_questions(survey_json):
    """Enhanced question extraction with complete structure"""
    questions = []
    global_position = 0
    
    for page_idx, page in enumerate(survey_json.get("pages", [])):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", "")
            family = question.get("family", "")
            
            if not q_text:
                continue
            
            global_position += 1
            
            # Enhanced schema type detection
            schema_type = determine_enhanced_schema_type(question)
            
            # Quality scoring
            quality_score = enhanced_quality_score(q_text)
            
            # Identity detection
            is_identity, identity_type = detect_identity_question(q_text)
            
            # Base question record
            question_record = {
                "question_text": q_text,
                "question_id": q_id,
                "position": global_position,
                "page_number": page_idx + 1,
                "is_choice": False,
                "parent_question": None,
                "schema_type": schema_type,
                "family": family,
                "quality_score": quality_score,
                "is_identity": is_identity,
                "identity_type": identity_type,
                "survey_id": survey_json.get("id", ""),
                "survey_title": survey_json.get("title", ""),
                "choices_count": 0,
                "has_other_option": False
            }
            
            # Extract choices with enhanced information
            choices = question.get("answers", {}).get("choices", [])
            question_record["choices_count"] = len(choices)
            
            # Check for "Other" option
            question_record["has_other_option"] = any(
                "other" in choice.get("text", "").lower() 
                for choice in choices
            )
            
            questions.append(question_record)
            
            # Add choice records
            for choice_idx, choice in enumerate(choices):
                choice_text = choice.get("text", "")
                if choice_text:
                    choice_record = question_record.copy()
                    choice_record.update({
                        "question_text": f"{q_text} - {choice_text}",
                        "choice_text": choice_text,
                        "choice_id": choice.get("id", ""),
                        "choice_position": choice_idx + 1,
                        "is_choice": True,
                        "parent_question": q_text,
                        "quality_score": enhanced_quality_score(choice_text),
                        "is_identity": False,  # Choices are not identity questions
                        "identity_type": None
                    })
                    questions.append(choice_record)
    
    return questions

def determine_enhanced_schema_type(question):
    """Enhanced schema type determination"""
    family = question.get("family", "")
    choices = question.get("answers", {}).get("choices", [])
    
    if family == "single_choice":
        return "Single Choice"
    elif family == "multiple_choice":
        return "Multiple Choice"
    elif family == "open_ended":
        return "Open-Ended"
    elif family == "matrix":
        return "Matrix"
    elif family == "ranking":
        return "Ranking"
    else:
        # Fallback logic
        if choices:
            if len(choices) <= 2:
                return "Single Choice"
            else:
                return "Multiple Choice"
        else:
            return "Open-Ended"

# ============= SNOWFLAKE ENHANCED FUNCTIONS =============
@st.cache_data(ttl=900)
def get_enhanced_snowflake_references():
    """Enhanced Snowflake reference data with authority counts"""
    query = """
    SELECT 
        HEADING_0, 
        UID, 
        COUNT(*) as AUTHORITY_COUNT,
        MAX(CREATED_AT) as LATEST_USAGE
    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
    WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL 
    AND TRIM(HEADING_0) != ''
    GROUP BY HEADING_0, UID
    ORDER BY UID, AUTHORITY_COUNT DESC
    """
    
    try:
        with get_enhanced_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        
        result.columns = result.columns.str.upper()
        
        # Add UID Final reference
        result['UID_FINAL'] = result['HEADING_0'].apply(
            lambda x: UID_FINAL_REFERENCE.get(x, None)
        )
        
        logger.info(f"Enhanced Snowflake references loaded: {len(result)} records")
        return result
        
    except Exception as e:
        logger.error(f"Failed to load enhanced Snowflake references: {e}")
        return pd.DataFrame()

# ============= ENHANCED MATCHING FUNCTIONS =============
@st.cache_data(ttl=1800)
def compute_enhanced_tfidf_matches(reference_df, target_df):
    """Enhanced TF-IDF matching with improved preprocessing"""
    if reference_df.empty or target_df.empty:
        return target_df
    
    # Enhanced normalization
    ref_texts = reference_df["HEADING_0"].apply(enhanced_normalize_text).tolist()
    target_texts = target_df["question_text"].apply(enhanced_normalize_text).tolist()
    
    # TF-IDF with enhanced parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        stop_words='english'
    )
    
    ref_vectors = vectorizer.fit_transform(ref_texts)
    target_vectors = vectorizer.transform(target_texts)
    
    similarity_matrix = cosine_similarity(target_vectors, ref_vectors)
    
    # Enhanced matching logic
    matched_results = []
    for i, similarities in enumerate(similarity_matrix):
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score >= TFIDF_HIGH_CONFIDENCE:
            confidence = "✅ High"
        elif best_score >= TFIDF_LOW_CONFIDENCE:
            confidence = "⚠️ Medium"
        else:
            confidence = "❌ Low"
            best_idx = None
        
        if best_idx is not None:
            matched_uid = reference_df.iloc[best_idx]["UID"]
            matched_question = reference_df.iloc[best_idx]["HEADING_0"]
            authority_count = reference_df.iloc[best_idx].get("AUTHORITY_COUNT", 1)
        else:
            matched_uid = None
            matched_question = None
            authority_count = 0
        
        matched_results.append({
            "tfidf_uid": matched_uid,
            "tfidf_question": matched_question,
            "tfidf_score": round(best_score, 4),
            "tfidf_confidence": confidence,
            "tfidf_authority": authority_count
        })
    
    # Add results to target dataframe
    for i, result in enumerate(matched_results):
        for key, value in result.items():
            target_df.loc[target_df.index[i], key] = value
    
    return target_df

@st.cache_data(ttl=1800)
def compute_enhanced_semantic_matches(reference_df, target_df):
    """Enhanced semantic matching with batch processing"""
    if reference_df.empty or target_df.empty:
        return target_df
    
    model = load_enhanced_model()
    
    # Batch processing for efficiency
    ref_embeddings = model.encode(
        reference_df["HEADING_0"].tolist(),
        batch_size=32,
        convert_to_tensor=True
    )
    
    target_embeddings = model.encode(
        target_df["question_text"].tolist(),
        batch_size=32,
        convert_to_tensor=True
    )
    
    # Compute similarities
    cosine_scores = util.cos_sim(target_embeddings, ref_embeddings)
    
    # Process results
    semantic_results = []
    for i in range(len(target_df)):
        best_idx = cosine_scores[i].argmax().item()
        best_score = cosine_scores[i][best_idx].item()
        
        if best_score >= SEMANTIC_THRESHOLD:
            matched_uid = reference_df.iloc[best_idx]["UID"]
            matched_question = reference_df.iloc[best_idx]["HEADING_0"]
            authority_count = reference_df.iloc[best_idx].get("AUTHORITY_COUNT", 1)
        else:
            matched_uid = None
            matched_question = None
            authority_count = 0
        
        semantic_results.append({
            "semantic_uid": matched_uid,
            "semantic_question": matched_question,
            "semantic_score": round(best_score, 4),
            "semantic_authority": authority_count
        })
    
    # Add results to target dataframe
    for i, result in enumerate(semantic_results):
        for key, value in result.items():
            target_df.loc[target_df.index[i], key] = value
    
    return target_df

def apply_enhanced_uid_final_matching(target_df):
    """Apply UID Final reference matching with enhanced logic"""
    target_df["uid_final_direct"] = None
    target_df["uid_final_match"] = False
    
    direct_matches = 0
    
    for idx, row in target_df.iterrows():
        question_text = row.get("question_text", "")
        
        # Direct match
        if question_text in UID_FINAL_REFERENCE:
            target_df.at[idx, "uid_final_direct"] = UID_FINAL_REFERENCE[question_text]
            target_df.at[idx, "uid_final_match"] = True
            direct_matches += 1
    
    logger.info(f"🎯 UID Final direct matches: {direct_matches}/{len(target_df)}")
    return target_df

def finalize_enhanced_matches(target_df):
    """Enhanced match finalization with priority logic"""
    # Apply UID Final reference first
    target_df = apply_enhanced_uid_final_matching(target_df)
    
    # Determine final UID with priority: UID Final > TF-IDF > Semantic
    def get_final_uid(row):
        if pd.notna(row.get("uid_final_direct")):
            return row["uid_final_direct"]
        elif pd.notna(row.get("tfidf_uid")) and row.get("tfidf_confidence") != "❌ Low":
            return row["tfidf_uid"]
        elif pd.notna(row.get("semantic_uid")):
            return row["semantic_uid"]
        else:
            return None
    
    target_df["final_uid"] = target_df.apply(get_final_uid, axis=1)
    
    # Convert UIDs to 3 characters
    uid_columns = ["final_uid", "tfidf_uid", "semantic_uid", "uid_final_direct"]
    for col in uid_columns:
        if col in target_df.columns:
            target_df[col] = target_df[col].apply(convert_uid_to_3_chars)
    
    # Handle identity questions (remove UID assignment)
    identity_mask = target_df.get("is_identity", False)
    target_df.loc[identity_mask, uid_columns] = None
    
    # Determine match type
    def get_match_type(row):
        if row.get("is_identity", False):
            return f"🔐 Identity: {row.get('identity_type', 'Unknown')}"
        elif pd.notna(row.get("uid_final_direct")):
            return "🎯 UID Final"
        elif pd.notna(row.get("tfidf_uid")) and row.get("tfidf_confidence") == "✅ High":
            return "✅ TF-IDF High"
        elif pd.notna(row.get("tfidf_uid")):
            return "⚠️ TF-IDF Medium"
        elif pd.notna(row.get("semantic_uid")):
            return "🧠 Semantic"
        else:
            return "❌ No Match"
    
    target_df["match_type"] = target_df.apply(get_match_type, axis=1)
    
    return target_df

def run_enhanced_uid_matching(reference_df, target_df):
    """Enhanced UID matching pipeline"""
    logger.info("🚀 Starting Enhanced UID Matching...")
    
    if reference_df.empty or target_df.empty:
        logger.warning("Empty dataframes provided for matching")
        return target_df
    
    # Process in batches for large datasets
    batch_size = BATCH_SIZE
    processed_batches = []
    
    for start_idx in range(0, len(target_df), batch_size):
        end_idx = min(start_idx + batch_size, len(target_df))
        batch_df = target_df.iloc[start_idx:end_idx].copy()
        
        logger.info(f"Processing batch {start_idx//batch_size + 1}: {len(batch_df)} questions")
        
        # Apply matching algorithms
        batch_df = compute_enhanced_tfidf_matches(reference_df, batch_df)
        batch_df = compute_enhanced_semantic_matches(reference_df, batch_df)
        batch_df = finalize_enhanced_matches(batch_df)
        
        processed_batches.append(batch_df)
    
    # Combine results
    final_result = pd.concat(processed_batches, ignore_index=True) if processed_batches else target_df
    
    logger.info("✅ Enhanced UID Matching completed")
    return final_result

# ============= ENHANCED EXPORT FUNCTIONS =============
def prepare_enhanced_export_data(final_df):
    """Enhanced export preparation with dual-table structure"""
    if final_df is None or final_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter main questions only
    main_questions = final_df[final_df["is_choice"] == False].copy()
    
    if main_questions.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Split into identity and non-identity
    identity_questions = main_questions[main_questions.get("is_identity", False) == True].copy()
    non_identity_questions = main_questions[main_questions.get("is_identity", False) == False].copy()
    
    # Prepare non-identity export
    export_non_identity = pd.DataFrame()
    if not non_identity_questions.empty:
        export_non_identity = non_identity_questions[[
            'survey_id', 'question_id', 'position', 'final_uid'
        ]].copy()
        export_non_identity.columns = ['SURVEY_ID', 'QUESTION_ID', 'QUESTION_NUMBER', 'UID']
    
    # Prepare identity export (no UID column)
    export_identity = pd.DataFrame()
    if not identity_questions.empty:
        identity_questions['row_id'] = range(1, len(identity_questions) + 1)
        export_identity = identity_questions[[
            'survey_id', 'question_id', 'row_id', 'position', 'identity_type'
        ]].copy()
        export_identity.columns = ['SURVEY_ID', 'QUESTION_ID', 'ROW_ID', 'QUESTION_NUMBER', 'IDENTITY_TYPE']
    
    return export_non_identity, export_identity

# ============= MAIN APPLICATION =============
def main():
    """Enhanced main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🚀 Enhanced UID Matcher V11</div>', unsafe_allow_html=True)
    st.markdown('<div class="hybrid-info">🔄 <strong>Hybrid Approach:</strong> SurveyMonkey Complete Question Bank + Snowflake UID References</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🚀 Enhanced Navigation")
        
        pages = {
            "🏠 Enhanced Home": "enhanced_home",
            "📊 Question Bank Builder": "question_bank_builder", 
            "🔧 Enhanced UID Matching": "enhanced_matching",
            "📥 Enhanced Export": "enhanced_export",
            "📊 Analytics Dashboard": "analytics"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, use_container_width=True):
                st.session_state.page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Connection status
        st.markdown("**🔗 Enhanced Status**")
        try:
            token = get_enhanced_surveymonkey_token()
            sm_status = "✅" if token else "❌"
        except:
            sm_status = "❌"
        
        try:
            get_enhanced_snowflake_engine()
            sf_status = "✅"
        except:
            sf_status = "❌"
        
        st.write(f"📊 SurveyMonkey: {sm_status}")
        st.write(f"❄️ Snowflake: {sf_status}")
        st.write(f"🎯 UID Final Refs: {len(UID_FINAL_REFERENCE)}")
    
    # Page routing
    page = st.session_state.get("page", "enhanced_home")
    
    if page == "enhanced_home":
        show_enhanced_home()
    elif page == "question_bank_builder":
        show_question_bank_builder()
    elif page == "enhanced_matching":
        show_enhanced_matching()
    elif page == "enhanced_export":
        show_enhanced_export()
    elif page == "analytics":
        show_analytics_dashboard()
    else:
        st.error("❌ Unknown page")

def show_enhanced_home():
    """Enhanced home page"""
    st.markdown("## 🏠 Enhanced UID Matcher V11")
    
    # Key improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Hybrid Approach Benefits")
        st.markdown("• **Complete Question Structure** from SurveyMonkey")
        st.markdown("• **Authoritative UID References** from Snowflake")
        st.markdown("• **Enhanced Identity Detection** (21 types)")
        st.markdown("• **Improved Quality Scoring** algorithm")
        st.markdown("• **3-Character UID Conversion** applied")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Enhanced Features")
        st.markdown("• **Batch Processing** for performance")
        st.markdown("• **Multi-Algorithm Matching** (TF-IDF + Semantic + UID Final)")
        st.markdown("• **Dual Export Tables** (Identity vs Non-Identity)")
        st.markdown("• **Advanced Caching** for speed")
        st.markdown("• **Comprehensive Analytics** dashboard")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        surveys = get_enhanced_surveys_from_2023()
        st.metric("📊 Available Surveys", len(surveys))
    
    with col2:
        try:
            sf_refs = get_enhanced_snowflake_references()
            st.metric("❄️ Snowflake References", len(sf_refs))
        except:
            st.metric("❄️ Snowflake References", "Error")
    
    with col3:
        st.metric("🎯 UID Final References", len(UID_FINAL_REFERENCE))
    
    with col4:
        identity_types = len(IDENTITY_TYPES)
        st.metric("🔐 Identity Types", identity_types)
    
    # Workflow
    st.markdown("### 🚀 Enhanced Workflow")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Build Question Bank")
        st.markdown("Extract complete question structures from SurveyMonkey")
        if st.button("📊 Start Building", use_container_width=True):
            st.session_state.page = "question_bank_builder"
            st.rerun()
    
    with col2:
        st.markdown("#### 2️⃣ Enhanced Matching")
        st.markdown("Apply hybrid matching algorithms")
        if st.button("🔧 Start Matching", use_container_width=True):
            st.session_state.page = "enhanced_matching"
            st.rerun()
    
    with col3:
        st.markdown("#### 3️⃣ Enhanced Export")
        st.markdown("Export dual-table structure")
        if st.button("📥 Export Data", use_container_width=True):
            st.session_state.page = "enhanced_export"
            st.rerun()

def show_question_bank_builder():
    """Enhanced question bank builder with automatic 2023+ processing and advanced deduplication"""
    st.markdown("## 📊 Enhanced Question Bank Builder")
    st.markdown('<div class="question-bank-card">🤖 <strong>AI-Powered Deduplication:</strong> Automatically processes all surveys from 2023+ with semantic similarity and fuzzy matching</div>', unsafe_allow_html=True)
    
    # Information panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 **Automated Process:**
        - ✅ **Auto-fetch** all surveys from 2023 onwards
        - 🤖 **AI Deduplication** using sentence transformers  
        - 🔤 **Fuzzy matching** for text similarity
        - 📊 **Quality scoring** for best question selection
        - 🔐 **Identity detection** with 21 types
        """)
    
    with col2:
        # Show current stats if question bank exists
        if st.session_state.question_bank_complete is not None:
            df = st.session_state.question_bank_complete
            st.metric("📊 Total Questions", len(df))
            st.metric("❓ Main Questions", len(df[df["is_choice"] == False]))
            st.metric("🔘 Choices", len(df[df["is_choice"] == True]))
    
    # Build question bank button
    if st.button("🚀 Build AI-Powered Question Bank", type="primary", help="Process all surveys from 2023+ with advanced deduplication"):
        
        # Get surveys from 2023+
        surveys_2023 = get_enhanced_surveys_from_2023()
        
        if not surveys_2023:
            st.error("❌ No surveys found from 2023 onwards or connection failed")
            return
        
        st.info(f"📅 Found {len(surveys_2023)} surveys from 2023 onwards")
        
        all_questions = []
        token = get_enhanced_surveymonkey_token()
        
        if not token:
            st.error("❌ SurveyMonkey token not available")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each survey
        for i, survey in enumerate(surveys_2023):
            survey_id = survey["id"]
            survey_title = survey.get("title", "Unknown")
            
            status_text.text(f"🔍 Processing survey {i+1}/{len(surveys_2023)}: {survey_title[:50]}...")
            
            try:
                # Get survey details
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(
                    f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details", 
                    headers=headers
                )
                response.raise_for_status()
                survey_json = response.json()
                
                # Extract questions with enhanced structure
                questions = extract_enhanced_questions(survey_json)
                all_questions.extend(questions)
                
                time.sleep(REQUEST_DELAY)
                progress_bar.progress((i + 1) / len(surveys_2023))
                
            except Exception as e:
                st.warning(f"⚠️ Skipped survey {survey_id}: {str(e)[:100]}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_questions:
            st.error("❌ No questions extracted from surveys")
            return
        
        # Create initial dataframe
        st.info(f"📊 Extracted {len(all_questions)} total questions and choices")
        all_questions_df = pd.DataFrame(all_questions)
        
        # Advanced deduplication
        dedup_df = deduplicate_questions_advanced(all_questions_df)
        
        # Save to session state
        st.session_state.question_bank_complete = dedup_df
        
        # Success summary
        st.success(f"✅ **Question Bank Created Successfully!**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        main_questions = dedup_df[dedup_df["is_choice"] == False]
        choices = dedup_df[dedup_df["is_choice"] == True]
        identity_questions = dedup_df[dedup_df["is_identity"] == True]
        high_quality = dedup_df[dedup_df["quality_score"] >= QUESTION_QUALITY_THRESHOLD]
        
        with col1:
            st.metric("❓ Main Questions", len(main_questions))
        with col2:
            st.metric("🔘 Choice Options", len(choices))
        with col3:
            st.metric("🔐 Identity Questions", len(identity_questions))
        with col4:
            st.metric("⭐ High Quality", len(high_quality))
    
    # Display current question bank
    if st.session_state.question_bank_complete is not None:
        df = st.session_state.question_bank_complete
        
        st.markdown("---")
        st.markdown("### 📊 Question Bank Explorer")
        
        # Simple filter dropdown as requested
        col1, col2 = st.columns([1, 2])
        
        with col1:
            question_filter = st.selectbox(
                "📋 Question Type Filter:",
                ["All", "Main Question", "Choice"],
                key="qb_simple_filter"
            )
        
        with col2:
            # Search box
            search_term = st.text_input("🔍 Search questions:", key="qb_search")
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply question type filter
        if question_filter == "Main Question":
            filtered_df = filtered_df[filtered_df["is_choice"] == False]
        elif question_filter == "Choice":
            filtered_df = filtered_df[filtered_df["is_choice"] == True]
        
        # Apply search filter
        if search_term:
            mask = filtered_df["question_text"].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Display results
        if not filtered_df.empty:
            # Results summary
            st.info(f"📊 Showing {len(filtered_df)} results")
            
            # Display columns
            display_columns = [
                "question_text", "schema_type", "quality_score", "is_identity", 
                "identity_type", "choices_count", "survey_title"
            ]
            
            st.dataframe(
                filtered_df[display_columns],
                column_config={
                    "question_text": st.column_config.TextColumn("Question Text", width="large"),
                    "schema_type": st.column_config.TextColumn("Type", width="medium"),
                    "quality_score": st.column_config.NumberColumn("Quality", width="small", format="%.2f"),
                    "is_identity": st.column_config.CheckboxColumn("Identity?", width="small"),
                    "identity_type": st.column_config.TextColumn("Identity Type", width="medium"),
                    "choices_count": st.column_config.NumberColumn("Choices", width="small"),
                    "survey_title": st.column_config.TextColumn("Survey", width="medium")
                },
                use_container_width=True,
                height=500
            )
            
            # Download option
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Question Bank",
                csv_data,
                f"question_bank_{question_filter.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("ℹ️ No questions match the selected filters")

def show_enhanced_matching():
    """Enhanced UID matching page"""
    st.markdown("## 🔧 Enhanced UID Matching")
    st.markdown('<div class="question-bank-card">🎯 <strong>Multi-Algorithm Matching:</strong> TF-IDF + Semantic + UID Final Reference</div>', unsafe_allow_html=True)
    
    # Check if question bank is available
    if st.session_state.question_bank_complete is None:
        st.warning("⚠️ No question bank available. Please build question bank first.")
        if st.button("📊 Go to Question Bank Builder"):
            st.session_state.page = "question_bank_builder"
            st.rerun()
        return
    
    # Load Snowflake references
    try:
        with st.spinner("📊 Loading Snowflake references..."):
            snowflake_refs = get_enhanced_snowflake_references()
            st.session_state.snowflake_references = snowflake_refs
    except Exception as e:
        st.error(f"❌ Failed to load Snowflake references: {e}")
        snowflake_refs = pd.DataFrame()
    
    if snowflake_refs.empty:
        st.warning("⚠️ No Snowflake references available. Matching will be limited.")
    
    # Show current data stats
    question_bank = st.session_state.question_bank_complete
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        main_q = len(question_bank[question_bank["is_choice"] == False])
        st.metric("📊 Questions to Match", main_q)
    
    with col2:
        st.metric("❄️ Snowflake References", len(snowflake_refs))
    
    with col3:
        st.metric("🎯 UID Final References", len(UID_FINAL_REFERENCE))
    
    with col4:
        identity_q = len(question_bank[question_bank["is_identity"] == True])
        st.metric("🔐 Identity Questions", identity_q)
    
    # Matching configuration
    st.markdown("### ⚙️ Matching Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        match_main_only = st.checkbox("Match main questions only", value=True)
    with col2:
        exclude_identity = st.checkbox("Exclude identity questions from UID assignment", value=True)
    with col3:
        min_quality = st.slider("Minimum quality score", 0, 100, QUESTION_QUALITY_THRESHOLD)
    
    # Run matching
    if st.button("🚀 Run Enhanced UID Matching", type="primary"):
        
        # Prepare target questions
        target_questions = question_bank.copy()
        
        if match_main_only:
            target_questions = target_questions[target_questions["is_choice"] == False]
        
        if min_quality > 0:
            target_questions = target_questions[target_questions["quality_score"] >= min_quality]
        
        if target_questions.empty:
            st.error("❌ No questions to match with current filters")
            return
        
        st.info(f"🎯 Matching {len(target_questions)} questions...")
        
        # Run enhanced matching
        with st.spinner("🔄 Running enhanced UID matching..."):
            matched_results = run_enhanced_uid_matching(snowflake_refs, target_questions)
            st.session_state.matched_results = matched_results
        
        st.success("✅ Enhanced UID matching completed!")
    
    # Display results
    if st.session_state.matched_results is not None:
        results_df = st.session_state.matched_results
        
        st.markdown("### 📊 Matching Results")
        
        # Results statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            uid_final_matches = results_df["uid_final_match"].sum()
            st.metric("🎯 UID Final Matches", uid_final_matches)
        
        with col2:
            high_tfidf = len(results_df[results_df["tfidf_confidence"] == "✅ High"])
            st.metric("✅ High TF-IDF", high_tfidf)
        
        with col3:
            semantic_matches = results_df["semantic_uid"].notna().sum()
            st.metric("🧠 Semantic Matches", semantic_matches)
        
        with col4:
            total_matched = results_df["final_uid"].notna().sum()
            match_rate = (total_matched / len(results_df) * 100) if len(results_df) > 0 else 0
            st.metric("📈 Overall Match Rate", f"{match_rate:.1f}%")
        
        # Results table with enhanced filters
        st.markdown("#### 🔍 Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_type_filter = st.multiselect(
                "Match Type:",
                results_df["match_type"].unique().tolist(),
                default=results_df["match_type"].unique().tolist(),
                key="match_type_filter"
            )
        
        with col2:
            schema_type_filter = st.multiselect(
                "Schema Type:",
                results_df["schema_type"].unique().tolist(),
                default=results_df["schema_type"].unique().tolist(),
                key="schema_type_filter"
            )
        
        with col3:
            survey_filter = st.multiselect(
                "Survey:",
                results_df["survey_title"].unique().tolist(),
                default=results_df["survey_title"].unique().tolist()[:3],
                key="survey_filter"
            )
        
        # Apply filters
        filtered_results = results_df.copy()
        
        if match_type_filter:
            filtered_results = filtered_results[filtered_results["match_type"].isin(match_type_filter)]
        
        if schema_type_filter:
            filtered_results = filtered_results[filtered_results["schema_type"].isin(schema_type_filter)]
        
        if survey_filter:
            filtered_results = filtered_results[filtered_results["survey_title"].isin(survey_filter)]
        
        # Display results table
        if not filtered_results.empty:
            display_columns = [
                "question_text", "final_uid", "match_type", "tfidf_score", 
                "semantic_score", "quality_score", "schema_type", "survey_title"
            ]
            
            st.dataframe(
                filtered_results[display_columns],
                column_config={
                    "question_text": st.column_config.TextColumn("Question", width="large"),
                    "final_uid": st.column_config.TextColumn("Final UID", width="small"),
                    "match_type": st.column_config.TextColumn("Match Type", width="medium"),
                    "tfidf_score": st.column_config.NumberColumn("TF-IDF Score", width="small", format="%.3f"),
                    "semantic_score": st.column_config.NumberColumn("Semantic Score", width="small", format="%.3f"),
                    "quality_score": st.column_config.NumberColumn("Quality", width="small"),
                    "schema_type": st.column_config.TextColumn("Type", width="medium"),
                    "survey_title": st.column_config.TextColumn("Survey", width="medium")
                },
                use_container_width=True,
                height=500
            )
            
            # Download results
            csv_results = filtered_results.to_csv(index=False)
            st.download_button(
                "📥 Download Matching Results",
                csv_results,
                f"enhanced_matching_results_{uuid4().hex[:8]}.csv",
                "text/csv"
            )
        else:
            st.info("ℹ️ No results match the selected filters")

def show_enhanced_export():
    """Enhanced export page"""
    st.markdown("## 📥 Enhanced Export")
    st.markdown('<div class="question-bank-card">📋 <strong>Dual Table Export:</strong> Identity questions (no UID) + Non-identity questions (with UID)</div>', unsafe_allow_html=True)
    
    if st.session_state.matched_results is None:
        st.warning("⚠️ No matching results available. Please run UID matching first.")
        if st.button("🔧 Go to Enhanced Matching"):
            st.session_state.page = "enhanced_matching"
            st.rerun()
        return
    
    # Prepare export data
    results_df = st.session_state.matched_results
    
    with st.spinner("📊 Preparing enhanced export data..."):
        export_non_identity, export_identity = prepare_enhanced_export_data(results_df)
    
    # Export statistics
    st.markdown("### 📊 Export Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Non-Identity Questions", len(export_non_identity))
    
    with col2:
        st.metric("🔐 Identity Questions", len(export_identity))
    
    with col3:
        total_records = len(export_non_identity) + len(export_identity)
        st.metric("📋 Total Export Records", total_records)
    
    with col4:
        uid_assigned = export_non_identity["UID"].notna().sum() if not export_non_identity.empty else 0
        st.metric("🎯 UIDs Assigned", uid_assigned)
    
    # Preview tables
    st.markdown("### 👁️ Export Preview")
    
    tab1, tab2 = st.tabs(["📊 Non-Identity Table", "🔐 Identity Table"])
    
    with tab1:
        st.markdown("**Structure:** SURVEY_ID, QUESTION_ID, QUESTION_NUMBER, UID")
        if not export_non_identity.empty:
            st.dataframe(
                export_non_identity.head(20),
                column_config={
                    "SURVEY_ID": st.column_config.TextColumn("Survey ID", width="medium"),
                    "QUESTION_ID": st.column_config.TextColumn("Question ID", width="medium"),
                    "QUESTION_NUMBER": st.column_config.NumberColumn("Question #", width="small"),
                    "UID": st.column_config.TextColumn("UID", width="small")
                },
                use_container_width=True
            )
        else:
            st.info("ℹ️ No non-identity questions to export")
    
    with tab2:
        st.markdown("**Structure:** SURVEY_ID, QUESTION_ID, ROW_ID, QUESTION_NUMBER, IDENTITY_TYPE (No UID column)")
        if not export_identity.empty:
            st.dataframe(
                export_identity.head(20),
                column_config={
                    "SURVEY_ID": st.column_config.TextColumn("Survey ID", width="medium"),
                    "QUESTION_ID": st.column_config.TextColumn("Question ID", width="medium"),
                    "ROW_ID": st.column_config.NumberColumn("Row ID", width="small"),
                    "QUESTION_NUMBER": st.column_config.NumberColumn("Question #", width="small"),
                    "IDENTITY_TYPE": st.column_config.TextColumn("Identity Type", width="medium")
                },
                use_container_width=True
            )
        else:
            st.info("ℹ️ No identity questions to export")
    
    # Export actions
    st.markdown("### 🚀 Export Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not export_non_identity.empty:
            csv_non_identity = export_non_identity.to_csv(index=False)
            st.download_button(
                "📥 Download Non-Identity CSV",
                csv_non_identity,
                f"enhanced_non_identity_{uuid4().hex[:8]}.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col2:
        if not export_identity.empty:
            csv_identity = export_identity.to_csv(index=False)
            st.download_button(
                "📥 Download Identity CSV",
                csv_identity,
                f"enhanced_identity_{uuid4().hex[:8]}.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("🚀 Upload to Snowflake", use_container_width=True):
            try:
                with st.spinner("🚀 Uploading to Snowflake..."):
                    engine = get_enhanced_snowflake_engine()
                    
                    if not export_non_identity.empty:
                        table_name_1 = f"enhanced_non_identity_{uuid4().hex[:8]}"
                        export_non_identity.to_sql(
                            table_name_1, engine, if_exists='replace', 
                            index=False, method='multi'
                        )
                        st.success(f"✅ Non-identity table uploaded: {table_name_1}")
                    
                    if not export_identity.empty:
                        table_name_2 = f"enhanced_identity_{uuid4().hex[:8]}"
                        export_identity.to_sql(
                            table_name_2, engine, if_exists='replace', 
                            index=False, method='multi'
                        )
                        st.success(f"✅ Identity table uploaded: {table_name_2}")
                    
                    st.success("🎉 Enhanced export completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

def show_analytics_dashboard():
    """Analytics dashboard page"""
    st.markdown("## 📊 Enhanced Analytics Dashboard")
    st.markdown('<div class="question-bank-card">📈 <strong>Comprehensive Analytics:</strong> Question bank insights and matching performance</div>', unsafe_allow_html=True)
    
    # Check data availability
    has_question_bank = st.session_state.question_bank_complete is not None
    has_matching_results = st.session_state.matched_results is not None
    
    if not has_question_bank and not has_matching_results:
        st.warning("⚠️ No data available for analytics. Please build question bank and run matching first.")
        return
    
    # Question Bank Analytics
    if has_question_bank:
        st.markdown("### 📊 Question Bank Analytics")
        
        qb_df = st.session_state.question_bank_complete
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality distribution
            st.markdown("#### ⭐ Quality Score Distribution")
            quality_bins = pd.cut(qb_df['quality_score'], bins=[0, 30, 60, 80, 100], labels=['Low', 'Medium', 'High', 'Excellent'])
            quality_counts = quality_bins.value_counts()
            st.bar_chart(quality_counts)
        
        with col2:
            # Schema type distribution
            st.markdown("#### 📋 Question Type Distribution")
            schema_counts = qb_df['schema_type'].value_counts()
            st.bar_chart(schema_counts)
        
        # Identity analysis
        st.markdown("#### 🔐 Identity Question Analysis")
        
        identity_df = qb_df[qb_df['is_identity'] == True]
        if not identity_df.empty:
            identity_type_counts = identity_df['identity_type'].value_counts()
            st.bar_chart(identity_type_counts)
        else:
            st.info("ℹ️ No identity questions found")
    
    # Matching Results Analytics
    if has_matching_results:
        st.markdown("### 🎯 Matching Performance Analytics")
        
        results_df = st.session_state.matched_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Match type distribution
            st.markdown("#### 🔧 Match Type Distribution")
            match_type_counts = results_df['match_type'].value_counts()
            st.bar_chart(match_type_counts)
        
        with col2:
            # Score distributions
            st.markdown("#### 📊 Score Distributions")
            
            # TF-IDF scores
            tfidf_scores = results_df['tfidf_score'].dropna()
            if not tfidf_scores.empty:
                st.write("**TF-IDF Scores:**")
                st.line_chart(tfidf_scores.hist(bins=20, alpha=0.7))
            
            # Semantic scores  
            semantic_scores = results_df['semantic_score'].dropna()
            if not semantic_scores.empty:
                st.write("**Semantic Scores:**")
                st.line_chart(semantic_scores.hist(bins=20, alpha=0.7))
        
        # Performance summary
        st.markdown("#### 📈 Performance Summary")
        
        total_questions = len(results_df)
        matched_questions = results_df['final_uid'].notna().sum()
        match_rate = (matched_questions / total_questions * 100) if total_questions > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Questions", total_questions)
        
        with col2:
            st.metric("✅ Successfully Matched", matched_questions)
        
        with col3:
            st.metric("📈 Match Rate", f"{match_rate:.1f}%")
        
        with col4:
            avg_quality = results_df['quality_score'].mean()
            st.metric("⭐ Avg Quality Score", f"{avg_quality:.1f}")
    
    # Combined analytics
    if has_question_bank and has_matching_results:
        st.markdown("### 🔄 Combined Analytics")
        
        # Quality vs Match Success
        st.markdown("#### ⭐ Quality Score vs Match Success")
        
        qb_df = st.session_state.question_bank_complete
        results_df = st.session_state.matched_results
        
        # Merge data for analysis
        analysis_df = results_df[['question_id', 'final_uid', 'quality_score']].copy()
        analysis_df['has_uid'] = analysis_df['final_uid'].notna()
        
        # Group by quality ranges
        quality_ranges = pd.cut(analysis_df['quality_score'], bins=[0, 30, 60, 80, 100], labels=['0-30', '31-60', '61-80', '81-100'])
        quality_match_analysis = analysis_df.groupby(quality_ranges)['has_uid'].agg(['count', 'sum']).fillna(0)
        quality_match_analysis['match_rate'] = (quality_match_analysis['sum'] / quality_match_analysis['count'] * 100).fillna(0)
        
        st.bar_chart(quality_match_analysis['match_rate'])
        
        st.markdown("**Insight:** Higher quality questions tend to have better matching success rates.")

# ============= END OF APPLICATION =============

if __name__ == "__main__":
    main() 