import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import pandas as pd

# ============================================================================
# ENUMS - Replace magic strings
# ============================================================================

class AnalysisDecision(str, Enum):
    """Enum for analysis decisions"""
    GENERAL_CONVERSATION = "general_conversation"
    LOAD_SELECTED_TABLES = "load_selected_tables"
    ERROR_NO_TABLES = "error_no_tables"
    STOP_ANALYSIS = "stop_analysis"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    RELATED_ENTITIES = "related_entities"
    MASTER_DETAIL = "master_detail"
    COMPLEMENTARY_DATA = "complementary_data"
    SINGLE_ENTITY = "single_entity"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    EXPLORATORY = "exploratory"
    INTELLIGENT = "intelligent"
    FALLBACK = "fallback"


class RelationshipType(str, Enum):
    """Types of relationships between tables"""
    FOREIGN_KEY = "foreign_key"
    LOGICAL_BUSINESS = "logical_business"
    COMPLEMENTARY_ATTRIBUTES = "complementary_attributes"
    TIME_SERIES = "time_series"
    HIERARCHICAL = "hierarchical"
    NONE = "none"


class ConfidenceLevel(str, Enum):
    """Confidence levels for routing decisions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RoutingDecision:
    """Structured routing decision from LLM"""
    relevant_tables: List[str]
    analysis_type: AnalysisType
    relationship_type: RelationshipType
    confidence: ConfidenceLevel
    reasoning: str
    expected_insights: str


@dataclass
class TableMetadata:
    """Metadata for a single table"""
    name: str
    columns: List[str]
    data_types: List[str]
    row_count: int
    sample_data: Optional[pd.DataFrame] = None


@dataclass
class AnalysisContext:
    """Context for analysis operations"""
    user_query: str
    selected_tables: List[str]
    routing_decision: Optional[RoutingDecision]
    table_metadata: Dict[str, TableMetadata]

@dataclass
class AnalysisResult:
    """Result from analysis operation"""
    success: bool
    response: str
    selected_tables: List[str]
    analysis_type: AnalysisType
    metadata: Dict[str, Any]
    error: Optional[str] = None


# ============================================================================
# CONSTANTS
# ============================================================================

class AnalysisConfig:
    """Configuration constants"""
    MAX_TABLES_FOR_ROUTING = 4
    MAX_TABLES_FOR_ANALYSIS = 2
    LLM_DF_HEAD_ROWS = 5
    MAX_DISPLAY_ROWS = 150
    MAX_LLM_ITERATIONS = 3
    LLM_TIMEOUT_SECONDS = 30


# ============================================================================
# EXCEPTIONS
# ============================================================================

class DataAnalyzerError(Exception):
    """Base exception for data analyzer"""
    pass


class TableRoutingError(DataAnalyzerError):
    """Error during table routing"""
    pass

class LLMTimeoutError(DataAnalyzerError):
    """LLM operation timed out"""
    pass

class DataAnalysisError(DataAnalyzerError):
    """Error during data analysis"""
    pass


class SecurityError(DataAnalyzerError):
    """Security-related error"""
    pass


# ============================================================================
# UTILITIES
# ============================================================================

class PandasOptionsManager:
    """Context manager for pandas display options"""
    
    def __init__(self, max_rows: Optional[int] = None):
        self.max_rows = max_rows
        self.original_settings = {}
    
    def __enter__(self):
        self.original_settings = {
            'display.max_columns': pd.get_option('display.max_columns'),
            'display.max_colwidth': pd.get_option('display.max_colwidth'),
            'display.width': pd.get_option('display.width'),
            'display.max_rows': pd.get_option('display.max_rows'),
            'display.expand_frame_repr': pd.get_option('display.expand_frame_repr'),
        }
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', self.max_rows)
        pd.set_option('display.expand_frame_repr', False)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for option, value in self.original_settings.items():
            pd.set_option(option, value)


class QuerySanitizer:
    """Sanitize user queries for security"""
    
    @staticmethod
    def sanitize(query: str) -> str:
        """Remove potentially dangerous content from query"""
        dangerous_patterns = [
            'import ', '__import__', 'exec(', 'eval(', 
            'os.', 'sys.', 'subprocess', '__builtins__',
            'open(', 'file(', 'input(', 'raw_input('
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in query_lower:
                raise SecurityError(f"Query contains potentially dangerous pattern: {pattern}")
        
        return query.strip()


class ResponseCleaner:
    """Clean LLM agent responses"""
    
    SKIP_PATTERNS = [
        'Thought:', 'Action:', 'Action Input:', 'Observation:',
        '> Entering new', '> Finished chain',
        'I need to', 'Let me', 'I should', 'I will'
    ]
    
    @classmethod
    def clean(cls, output: str, table_names: Optional[List[str]] = None) -> str:
        """Clean agent output removing chain-of-thought artifacts"""
        try:
            lines = output.split('\n')
            cleaned_lines = []
            in_final_answer = False
            
            for line in lines:
                if line.strip().startswith('Final Answer:'):
                    in_final_answer = True
                    after_colon = line.split(':', 1)[1].strip() if ':' in line else ''
                    if after_colon:
                        cleaned_lines.append(after_colon)
                    continue
                
                if in_final_answer or cls._is_data_line(line):
                    cleaned_line = cls._remove_table_references(line, table_names or [])
                    if cleaned_line.strip():
                        cleaned_lines.append(cleaned_line)
                    continue
                
                if any(pattern in line for pattern in cls.SKIP_PATTERNS):
                    continue
                
                if line.strip():
                    cleaned_lines.append(line)
            
            result = '\n'.join(cleaned_lines).strip()
            result = result.replace('...', '')
            result = cls._normalize_whitespace(result)
            
            return result if len(result) > 30 else output
            
        except Exception as e:
            logging.warning(f"Error cleaning output: {e}")
            return output
    
    @staticmethod
    def _is_data_line(line: str) -> bool:
        return '|' in line or 'dtype:' in line or line.count(' ') > 10
    
    @staticmethod
    def _remove_table_references(line: str, table_names: List[str]) -> str:
        for table_name in table_names:
            patterns = [
                f"From {table_name}:",
                f"**From {table_name}:**",
                f"Based on {table_name}:",
            ]
            for pattern in patterns:
                line = line.replace(pattern, "")
        return line
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        import re
        return re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
