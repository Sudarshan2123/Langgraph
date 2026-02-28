


import logging
from typing import Dict, List
import pandas as pd 
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from entity.core import AnalysisDecision, TableMetadata
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

class DataAnalyzer:
    """
    Main data analyzer class with proper separation of concerns.
    Orchestrates routing and analysis operations.
    """
    
    def __init__(self, config, vertex_llm, engine, enable_naturalization=True):
        """Initialize with optional response naturalization"""
        try:
            
            self.config = config
            
            # Initialize LLM with optimized settings
            self.vertex_llm = vertex_llm
            
            # Initialize components
            # self.router = TableRouter(self.llm)

            # 🔧 NEW: Initialize Database Toolkit
            sql_engine = engine
            db = SQLDatabase(
                engine=sql_engine,
                schema="public",
                include_tables=None  # Allow dynamic table discovery
            )
            
            self.toolkit = SQLDatabaseToolkit(db=db, llm=self.vertex_llm)
            self.tools = self.toolkit.get_tools()
            
            logging.info(f"Initialized {len(self.tools)} database interaction tools")
            
            # NEW: Initialize naturalizer
            self.enable_naturalization = enable_naturalization
            if enable_naturalization:
                # self.naturalizer = ResponseNaturalizer(self.llm)
                logging.info("Response naturalization enabled")
            else:
                self.naturalizer = None
            
            logging.info("DataAnalyzer (SQL Mode) initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize DataAnalyzer: {e}")
            raise
    
    def detect_table_intent(self, state: dict) -> dict:
        """
        Detect which tables are relevant for the user's query.
        
        Args:
            state: Agent state with user input and available tables
            
        Returns:
            Updated state with routing decision
        """
        user_input = state.get("input", "")
        available_tables = state.get("available_tables", {})
        
        if not available_tables:
            logging.error("No tables available for routing")
            return {
                **state,
                "decision": AnalysisDecision.ERROR_NO_TABLES.value,
                "response": "No database tables are currently loaded."
            }
        
        try:
            # Convert metadata format
            table_metadata = {
                name: TableMetadata(
                    name=name,
                    columns=meta.get("columns", []),
                    data_types=meta.get("data_types", []),
                    row_count=meta.get("row_count", 0)
                )
                for name, meta in available_tables.items()
            }
            
            # Route tables
            routing_decision = self.router.route_tables(user_input, table_metadata)
            
            logging.info(f"Routed to tables: {routing_decision.relevant_tables}")
            
            return {
                **state,
                "selected_tables": routing_decision.relevant_tables,
                "decision": AnalysisDecision.LOAD_SELECTED_TABLES.value,
                "analysis_context": {
                    "analysis_type": routing_decision.analysis_type.value,
                    "relationship_type": routing_decision.relationship_type.value,
                    "confidence": routing_decision.confidence.value,
                    "reasoning": routing_decision.reasoning,
                    "expected_insights": routing_decision.expected_insights
                }
            }
            
        except Exception as e:
            logging.error(f"Error in table routing: {e}", exc_info=True)
            return {
                **state,
                "selected_tables": [],
                "decision": AnalysisDecision.STOP_ANALYSIS.value,
                "analysis_context": f"Unable to determine relevant tables: {str(e)}"
            }
    def _create_lookup_aware_system_prompt(self, selected_tables: List[str]) -> str:
        """
        Create system prompt that enforces lookup table resolution.
        
        Args:
            selected_tables: List of tables selected by router
            
        Returns:
            Enhanced system prompt string
        """
        # Identify lookup/master tables
        lookup_tables = [t for t in selected_tables if 'master' in t.lower() or 'mst' in t.lower()]
        
        base_prompt = f"""You are an expert SQL analyst working with an Oracle database.

    **AVAILABLE TABLES**: {', '.join(selected_tables)}

    **CRITICAL RULES**:

    1. **ALWAYS USE ALL AVAILABLE TABLES**:
    - You have been provided with {len(selected_tables)} carefully selected tables
    - Do NOT ignore any table, especially lookup/master tables
    - ALL tables are relevant to answering the user's query

    2. **LOOKUP TABLE RESOLUTION** (MANDATORY):
    - The following are lookup/master tables: {', '.join(lookup_tables) if lookup_tables else 'None identified'}
    - **ALWAYS JOIN** with lookup tables to resolve IDs to human-readable names
    - NEVER return raw ID values (designation_id, department_id, status_id, etc.)
    - Example: If you see 'designation_id' in employee_master, JOIN with 'designation_master'

    3. **SCHEMA INSPECTION**:
    - Call sql_db_schema for ALL tables: {', '.join(selected_tables)}
    - Examine foreign key relationships and column name patterns
    - Look for columns ending in '_id' or '_code' - these require lookups

    4. **QUERY CONSTRUCTION**:
    - Build comprehensive JOINs that include ALL relevant tables
    - Use LEFT JOIN for optional lookups, INNER JOIN for required data
    - Example pattern:
    ```sql
        SELECT 
            emp.emp_name,
            dept.dept_name,        -- NOT dept_id
            desig.designation_name -- NOT designation_id
        FROM employee_master emp
        LEFT JOIN department_mst dept ON emp.department_id = dept.department_id
        LEFT JOIN designation_master desig ON emp.designation_id = desig.designation_id
    ```

    5. **VALIDATION**:
    - Before executing, verify you've joined ALL lookup tables
    - If returning any column with '_id' or '_code' suffix, you've made an error

    6. **ERROR HANDLING**:
    - Oracle syntax: Do NOT use 'AS' keyword for table aliases
    - Correct: `FROM employee_master emp`
    - Incorrect: `FROM employee_master AS emp`

    **YOUR TASK**: Answer the user's query using ALL {len(selected_tables)} tables provided, 
    resolving ALL ID fields to human-readable values.
    
    **RESPONSE FORMAT**:
    - Provide COMPLETE information - do NOT truncate your response
    - If user asks for "table format", present data in a clear tabular structure
    - Include ALL relevant columns from the query results
    - Ensure your response is COMPLETE before finishing"""

        return base_prompt
    
    def _extract_clean_response(self, response: any) -> str:
        """
        Robustly extract clean text from various LangChain response formats.
        
        Handles:
        1. Simple strings (LangChain 0.1.x on localhost)
        2. Dicts with 'output' key (standard agent format)
        3. Lists of message chunks (LangChain 0.2.x + Vertex AI on server)
        4. AIMessage objects with .content attribute
        
        Args:
            response: Raw response from SQL agent executor
            
        Returns:
            Clean text string ready for naturalization
        """
        try:
            # Case 1: Already a clean string
            if isinstance(response, str):
                logging.debug("Response is already a string")
                return response.strip()
            
            # Case 2: Dictionary with 'output' key (most common)
            if isinstance(response, dict):
                if 'output' in response:
                    output = response['output']
                    logging.debug(f"Extracted 'output' key (type: {type(output)})")
                    return self._extract_clean_response(output)
                
                logging.warning(f"Dict without 'output' key. Keys: {list(response.keys())}")
                if 'result' in response:
                    return self._extract_clean_response(response['result'])
                return str(response)
            
            # Case 3: List of message chunks
            if isinstance(response, list):
                logging.debug(f"Response is a list with {len(response)} items")
                text_parts = []
                
                for idx, item in enumerate(response):
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                
                if text_parts:
                    clean_text = ' '.join(text_parts).strip()
                    logging.info(f"Extracted text from list: '{clean_text[:100]}...'")
                    return clean_text
                
                return str(response)
            
            # Case 4: LangChain message objects
            if hasattr(response, 'content'):
                logging.debug("Response has .content attribute")
                return self._extract_clean_response(response.content)
            
            # Case 5: Unknown format
            logging.warning(f"Unknown response type: {type(response)}")
            return str(response)
            
        except Exception as e:
            logging.error(f"Error in _extract_clean_response: {e}", exc_info=True)
            return str(response)
    
    def analyze_data_with_routing(self, state: dict) -> str:
        selected_tables = state.get("selected_tables", [])
        user_input = state.get("input", "")
        
        if not selected_tables:
            return "No tables selected for analysis."

        try:
            sql_engine = self.db_manager.get_sqlalchemy_engine()
            if sql_engine is None:
                return "Analysis error: Database engine is not available."
            
            db = SQLDatabase(
                engine=self.engine,
                schema="public",
                include_tables=selected_tables
            )

             # 🔧 FIX: Create enhanced system prompt that enforces lookup table usage
            agent_system_prompt = self._create_lookup_aware_system_prompt(selected_tables)
            
            logging.info(f"Creating SQL agent for tables: {selected_tables}")
            sql_agent_executor = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="openai-tools",
                verbose=True,  # Enable to debug incomplete responses
                handle_parsing_errors=True,
                prefix=agent_system_prompt,
                max_iterations=15,  # Increased for complex queries
                max_execution_time=90  # Increased timeout
            )
            
            raw_response_dict = sql_agent_executor.invoke({"input": user_input})
            
            # Debug logging
            logging.info(f"DEBUG - Agent output type: {type(raw_response_dict)}")
            if isinstance(raw_response_dict, dict):
                logging.info(f"DEBUG - Dict keys: {raw_response_dict.keys()}")
                logging.info(f"DEBUG - Full output: {raw_response_dict.get('output', 'NO OUTPUT KEY')}")
            
            # Extract response
            raw_response = self._extract_clean_response(raw_response_dict)
            
            logging.info(f"Extracted response length: {len(raw_response)} chars")
            logging.info(f"First 200 chars: {raw_response[:200]}")
            logging.info(f"Last 100 chars: {raw_response[-100:] if len(raw_response) > 100 else raw_response}")
            
            # Skip naturalization - it's not adding value and may truncate
            logging.info("Returning raw SQL agent response (naturalization disabled)")
            return raw_response
            
        except Exception as e:
            logging.error(f"Error in analysis: {e}", exc_info=True)
            return f"Analysis error: {str(e)}"

    def analyze_data_stream(
        self,
        dfs: Dict[str, pd.DataFrame],
        query: str
    ) -> str: # <- Changed return type from Generator to str
        """
        Legacy method for backward compatibility.
        Creates state and calls routing-based analysis.
        Returns a single response string.
        """
        # This would need session state from Streamlit
        # Better to inject dependencies instead
        logging.warning("analyze_data_stream called - prefer routing-based method")
        
        # Create minimal state
        state = {
            "input": query,
            "session_id": "legacy",
            "available_tables": {
                name: {
                    "columns": list(df.columns),
                    "data_types": [str(dt) for dt in df.dtypes],
                    "row_count": len(df)
                }
                for name, df in dfs.items()
            },
            "loaded_data": dfs
        }
        
        # Route tables
        state = self.detect_table_intent(state)
        
        # Check if routing failed
        if state.get("decision") != AnalysisDecision.LOAD_SELECTED_TABLES.value:
            return state.get("response", "Error during table routing.")

        # Analyze
        final_state = self.analyze_data_with_routing(state)
        
        # Return the final response string
        return final_state