import asyncio
import pickle
from typing import Any, Dict, List, Optional

from sqlalchemy import Engine, QueuePool, create_engine, text
from agentState import AgentState
from singleton import Init
import logging
import redis

class DataBase:
    def __init__(self):

        self.config = Init()
        self.engine: Optional[Engine] = None
        self.connection = None
        self.is_connected = False
        self.schema = None
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                username=self.config.REDIS_USERNAME,
                password=self.config.REDIS_PASSWORD,
                db=self.config.REDIS_DB,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.redis_enabled = True
            logging.info(f"Redis cache connected at {self.config.REDIS_HOST}:{self.config.REDIS_PORT}")
        except Exception as e:
            logging.warning(f"Redis unavailable: {e}. Using direct DB access.")
            self.redis_enabled = False
            self.redis_client = None

            self.cache_ttl = self.config.CACHE_TTL
        
        # Query result cache (in-memory for fast repeated queries)
        self._query_cache = {}
        self._query_cache_max_size = 1000

    def connect(self) -> bool:
        try:
            if not all([self.config.POSTGRES_USER, self.config.POSTGRES_PASSWORD, 
                       self.config.POSTGRES_HOST, self.config.POSTGRES_PORT, self.config.POSTGRES_DB]):
                logging.error("Database credentials missing")
                self.is_connected = False
                return False
            
            from urllib.parse import quote_plus
            encoded_password = quote_plus(self.config.POSTGRES_PASSWORD)
            engine_url = (
                f"postgresql+psycopg2://{self.config.POSTGRES_USER}:{encoded_password}@"
                f"{self.config.POSTGRES_HOST}:{self.config.POSTGRES_PORT}/{self.config.POSTGRES_DB}"
            )
            # Create the SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                engine_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=1800,
                pool_timeout=30,
                echo_pool=False
            )
            

            self.is_connected = True
            self.schema = 'public'
            logging.info(f"Connected to PostgreSQL '{self.config.POSTGRES_DB}' as '{self.config.POSTGRES_USER}'")
            return True
    
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {e}")
            self.is_connected = False
            return False


    def get_table_names(self) -> List[str]:
        """Get all table names from schema with caching"""
        if not self.engine:
            logging.error("No active database engine")
            return []
        
        # Check cache first
        cache_key = f"table_names:{self.config.POSTGRES_DB}"
        if self.redis_enabled:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    table_names = pickle.loads(cached)
                    logging.info(f"Table names from cache: {len(table_names)} tables")
                    return table_names
            except Exception as e:
                logging.warning(f"Cache read failed: {e}")
        
        try:
            query = (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                table_names = [row[0].lower() for row in result.fetchall()]
                logging.info(f"Retrieved {len(table_names)} table names from DB")
            
            # Cache for 1 hour
            if self.redis_enabled:
                try:
                    self.redis_client.setex(cache_key, 3600, pickle.dumps(table_names))
                except Exception as e:
                    logging.warning(f"Cache write failed: {e}")

            metadata_dict = {}
            for table_name in table_names:
                metadata_dict[table_name] = self.get_table_metadata(table_name)
            
            conn_data = {
                'analyzer': self.analyzer,
                'table_names': table_names,
                'table_metadata': metadata_dict,
                'loaded_data': {}, # This will stay empty!
                'schema': "public",
                'created_at': asyncio.get_event_loop().time()
            }

            global default_connection_data
            default_connection_data = conn_data

            return default_connection_data

        except Exception as e:
            logging.error(f"Error retrieving table names: {e}")
            return []

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get metadata for a specific table"""
        metadata = {
            'table_name': table_name,
            'columns': [],
            'data_types': [],
            'row_count': 0,
            'sample_data': {}
        }

        # Parameters to be used for all queries
        params = {
            'table_name': table_name.lower(),
            'schema': 'public'
        }
        
        try:
            with self.engine.connect() as conn:
                # 1. Get columns and data types
                col_query = text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name AND table_schema = :schema 
                    ORDER BY ordinal_position
                """)
                result1 = conn.execute(col_query, params)
                columns_info = result1.fetchall()
                
                metadata['columns'] = [col[0] for col in columns_info]
                metadata['data_types'] = [col[1] for col in columns_info]

                # 2. Get row count
                count_query = text(f"SELECT COUNT(*) FROM {params['schema']}.{params['table_name']}")
                result2 = conn.execute(count_query)
                row_count = result2.fetchone()
                metadata['row_count'] = row_count[0] if row_count and row_count[0] is not None else 0

                # 3. Get sample data
                if metadata['columns']:
                    sample_query = text(f"SELECT * FROM {params['schema']}.{params['table_name']} LIMIT 1")
                    result3 = conn.execute(sample_query) 
                    sample_row = result3.fetchone()
                    
                    if sample_row:
                        metadata['sample_data'] = dict(zip(metadata['columns'], sample_row))
            
            return metadata
    
        except Exception as e:
            # Note: If you want to use the ALL_TABLES query for row count, it must also use text():
            # count_query = text("SELECT num_rows FROM all_tables WHERE table_name = :table_name AND owner = :schema")
            logging.error(f"Error getting metadata for '{table_name}': {e}")
            return metadata

    