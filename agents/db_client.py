import os
import time
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
from functools import lru_cache

# Add SQLAlchemy support for relational databases
try:
    from sqlalchemy import create_engine, Column, Integer, Float, String, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.sql import text
    # Try to import asyncpg for PostgreSQL support
    import asyncpg
    SQLALCHEMY_AVAILABLE = True
    POSTGRESQL_AVAILABLE = True
except ImportError as e:
    SQLALCHEMY_AVAILABLE = False
    POSTGRESQL_AVAILABLE = False
    if 'sqlalchemy' in str(e):
        logging.warning("SQLAlchemy not available, PostgreSQL support disabled")
    else:
        logging.warning("SQLAlchemy asyncpg not available, PostgreSQL support may be limited")
        try:
            from sqlalchemy import create_engine, Column, Integer, Float, String, Text
            from sqlalchemy.ext.declarative import declarative_base
            from sqlalchemy.orm import sessionmaker
            SQLALCHEMY_AVAILABLE = True
        except ImportError:
            pass

logger = logging.getLogger(__name__)


class EventDB(ABC):
    @abstractmethod
    async def log_event(self, event_type: str, request_id: str, run_id: str = 'unknown', **kwargs) -> None:
        """Log an event to the database."""
        pass
    
    @abstractmethod
    async def get_events_by_request_id(self, request_id: str, run_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific request ID, optionally filtered by run_id."""
        pass
    
    @abstractmethod
    async def check_and_log_tool_call(self, normalized_call: str, function_name: str, arguments: Dict[str, Any], 
                                     request_id: str, branch_id: str = "main", run_id: str = "unknown") -> bool:
        """Check if a tool call exists and log it if not. Return True if it existed before."""
        pass
    
    @abstractmethod
    async def get_previous_tool_call(self, normalized_call: str) -> Optional[Dict[str, Any]]:
        """Get information about a previous occurrence of a tool call."""
        pass
    
    @abstractmethod
    async def get_events_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific run_id."""
        pass


class SQLiteEventDB(EventDB):
    def __init__(self, db_path: str = "./events.db"):
        self.db_path = db_path
        self._create_table()
    
    def _create_table(self):
        """Create the events and tool_calls tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create events table with request_id index for fast queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    event_type TEXT,
                    request_id TEXT,
                    run_id TEXT,
                    event_data TEXT
                )
            ''')
            
            # Create indexes for fast lookup
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_request_id ON events (request_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_run_id ON events (run_id)
            ''')
            
            # Create tool_calls table to track all unique tool calls across branches
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_call TEXT,
                    function_name TEXT,
                    arguments TEXT,
                    first_occurrence_time REAL,
                    first_occurrence_request_id TEXT,
                    first_occurrence_run_id TEXT,
                    occurrence_count INTEGER DEFAULT 1
                )
            ''')
            
            # Create unique index on normalized_call for fast lookups
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_calls_normalized ON tool_calls (normalized_call)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully initialized SQLite database tables at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database tables at {self.db_path}: {e}")
            raise
    
    async def log_event(self, event_type: str, request_id: str, run_id: str = 'unknown', **kwargs) -> None:
        """Log an event to SQLite database."""
        try:
            # Generate timestamp when the function is called, not when executed
            timestamp = time.time()
            event_data = json.dumps(kwargs, ensure_ascii=False)
            
            # Use synchronous sqlite3 in a thread-safe way
            def sync_log():
                conn = sqlite3.connect(self.db_path)
                with conn:
                    conn.execute(
                        "INSERT INTO events (timestamp, event_type, request_id, run_id, event_data) VALUES (?, ?, ?, ?, ?)",
                        (timestamp, event_type, request_id, run_id, event_data)
                    )
            
            # Run synchronous code in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sync_log)
            logger.info(f"Successfully logged event: {event_type} for request_id: {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    async def get_events_by_request_id(self, request_id: str, run_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific request ID, optionally filtered by run_id."""
        try:
            def sync_query():
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                with conn:
                    if run_id:
                        cursor = conn.execute(
                            "SELECT * FROM events WHERE request_id = ? AND run_id = ? ORDER BY timestamp",
                            (request_id, run_id)
                        )
                    else:
                        cursor = conn.execute(
                            "SELECT * FROM events WHERE request_id = ? ORDER BY timestamp",
                            (request_id,)
                        )
                    rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event_dict = dict(row)
                    event_dict['event_data'] = json.loads(event_dict['event_data'])
                    events.append(event_dict)
                
                return events
            
            # Run synchronous code in executor
            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, sync_query)
            logger.info(f"Successfully retrieved {len(events)} events for request_id: {request_id}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []
    
    async def check_and_log_tool_call(self, normalized_call: str, function_name: str, arguments: Dict[str, Any], 
                                     request_id: str, branch_id: str = "main", run_id: str = "unknown") -> bool:
        """Check if a tool call exists and log it if not. Return True if it existed before."""
        try:
            args_str = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            timestamp = time.time()
            
            def sync_check_and_log():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if the tool call already exists
                cursor.execute(
                    "SELECT id, first_occurrence_time, first_occurrence_request_id, first_occurrence_run_id FROM tool_calls WHERE normalized_call = ?",
                    (normalized_call,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Increment occurrence count
                    cursor.execute(
                        "UPDATE tool_calls SET occurrence_count = occurrence_count + 1 WHERE id = ?",
                        (existing[0],)
                    )
                    conn.commit()
                    conn.close()
                    return True, existing
                else:
                    # Insert new tool call
                    cursor.execute(
                        '''INSERT INTO tool_calls (normalized_call, function_name, arguments, first_occurrence_time, 
                           first_occurrence_request_id, first_occurrence_run_id) VALUES (?, ?, ?, ?, ?, ?)''',
                        (normalized_call, function_name, args_str, timestamp, request_id, run_id)
                    )
                    conn.commit()
                    conn.close()
                    return False, None
            
            # Run synchronous code in executor
            loop = asyncio.get_running_loop()
            existed, existing_record = await loop.run_in_executor(None, sync_check_and_log)
            
            if existed:
                logger.info(f"Successfully found and incremented existing tool call: {function_name} for normalized_call: {normalized_call}")
            else:
                logger.info(f"Successfully logged new tool call: {function_name} for normalized_call: {normalized_call}, request_id: {request_id}")
            
            return existed
            
        except Exception as e:
            logger.error(f"Failed to check and log tool call: {e}")
            return False
    
    async def get_previous_tool_call(self, normalized_call: str) -> Optional[Dict[str, Any]]:
        """Get information about a previous occurrence of a tool call."""
        try:
            def sync_query():
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                with conn:
                    cursor = conn.execute(
                        "SELECT * FROM tool_calls WHERE normalized_call = ?",
                        (normalized_call,)
                    )
                    row = cursor.fetchone()
                
                if row:
                    tool_call = dict(row)
                    tool_call['arguments'] = json.loads(tool_call['arguments'])
                    return tool_call
                return None
            
            # Run synchronous code in executor
            loop = asyncio.get_running_loop()
            tool_call = await loop.run_in_executor(None, sync_query)
            
            if tool_call:
                logger.info(f"Successfully retrieved previous tool call: {tool_call['function_name']} for normalized_call: {normalized_call}")
            else:
                logger.info(f"No previous tool call found for normalized_call: {normalized_call}")
            
            return tool_call
            
        except Exception as e:
            logger.error(f"Failed to retrieve previous tool call: {e}")
            return None
            
    async def get_events_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific run_id."""
        try:
            def sync_query():
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                with conn:
                    cursor = conn.execute(
                        "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp",
                        (run_id,)
                    )
                    rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event_dict = dict(row)
                    event_dict['event_data'] = json.loads(event_dict['event_data'])
                    events.append(event_dict)
                
                return events
            
            # Run synchronous code in executor
            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, sync_query)
            
            logger.info(f"Successfully retrieved {len(events)} events for run_id: {run_id}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events by run_id: {e}")
            return []


class DummyEventDB(EventDB):
    """Dummy implementation for testing purposes."""
    
    def __init__(self):
        self.events = []
        self.tool_calls = {}
    
    async def log_event(self, event_type: str, request_id: str, run_id: str = 'unknown', **kwargs) -> None:
        try:
            # Generate timestamp when the function is called
            timestamp = time.time()
            event = {
                'timestamp': timestamp,
                'event_type': event_type,
                'request_id': request_id,
                'run_id': run_id,
                'event_data': kwargs
            }
            self.events.append(event)
            logger.info(f"[DummyDB] Successfully logged event: {event_type} for request_id: {request_id}, run_id: {run_id}")
        except Exception as e:
            logger.error(f"[DummyDB] Failed to log event: {e}")
    
    async def get_events_by_request_id(self, request_id: str, run_id: str = None) -> List[Dict[str, Any]]:
        try:
            if run_id:
                events = [event for event in self.events if event['request_id'] == request_id and event.get('run_id') == run_id]
            else:
                events = [event for event in self.events if event['request_id'] == request_id]
            logger.info(f"[DummyDB] Successfully retrieved {len(events)} events for request_id: {request_id}{f', run_id: {run_id}' if run_id else ''}")
            return events
        except Exception as e:
            logger.error(f"[DummyDB] Failed to retrieve events: {e}")
            return []
    
    async def check_and_log_tool_call(self, normalized_call: str, function_name: str, arguments: Dict[str, Any], 
                                     request_id: str, branch_id: str = "main", run_id: str = "unknown") -> bool:
        """Check if a tool call exists and log it if not. Return True if it existed before."""
        try:
            if normalized_call in self.tool_calls:
                # Increment occurrence count
                self.tool_calls[normalized_call]['occurrence_count'] += 1
                logger.info(f"[DummyDB] Successfully found and incremented existing tool call: {function_name} for normalized_call: {normalized_call}")
                return True
            else:
                # Add new tool call
                self.tool_calls[normalized_call] = {
                    'normalized_call': normalized_call,
                    'function_name': function_name,
                    'arguments': arguments,
                    'first_occurrence_time': time.time(),
                    'first_occurrence_request_id': request_id,
                    'first_occurrence_run_id': run_id,
                    'occurrence_count': 1
                }
                logger.info(f"[DummyDB] Successfully logged new tool call: {function_name} for normalized_call: {normalized_call}, request_id: {request_id}, run_id: {run_id}")
                return False
        except Exception as e:
            logger.error(f"[DummyDB] Failed to check and log tool call: {e}")
            return False
    
    async def get_previous_tool_call(self, normalized_call: str) -> Optional[Dict[str, Any]]:
        """Get information about a previous occurrence of a tool call."""
        try:
            tool_call = self.tool_calls.get(normalized_call)
            if tool_call:
                logger.info(f"[DummyDB] Successfully retrieved previous tool call: {tool_call['function_name']} for normalized_call: {normalized_call}")
            else:
                logger.info(f"[DummyDB] No previous tool call found for normalized_call: {normalized_call}")
            return tool_call
        except Exception as e:
            logger.error(f"[DummyDB] Failed to retrieve previous tool call: {e}")
            return None
            
    async def get_events_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific run_id."""
        try:
            events = [event for event in self.events if event['run_id'] == run_id]
            logger.info(f"[DummyDB] Successfully retrieved {len(events)} events for run_id: {run_id}")
            return events
        except Exception as e:
            logger.error(f"[DummyDB] Failed to retrieve events by run_id: {e}")
            return []


if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class Event(Base):
        """SQLAlchemy model for events table."""
        __tablename__ = "events"
        
        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(Float, index=True)
        event_type = Column(String, index=True)
        request_id = Column(String, index=True)
        run_id = Column(String, index=True)
        event_data = Column(Text)
    
    class ToolCall(Base):
        """SQLAlchemy model for tool_calls table."""
        __tablename__ = "tool_calls"
        
        id = Column(Integer, primary_key=True, index=True)
        normalized_call = Column(String, unique=True, index=True)
        function_name = Column(String, index=True)
        arguments = Column(Text)
        first_occurrence_time = Column(Float, index=True)
        first_occurrence_request_id = Column(String, index=True)
        first_occurrence_run_id = Column(String, index=True)
        occurrence_count = Column(Integer, default=1)

    class SQLAlchemyEventDB(EventDB):
        """SQLAlchemy implementation supporting multiple databases via URL."""
        
        def __init__(self, db_url: str):
            self.db_url = db_url
            self._init_db()
        
        def _init_db(self):
            """Initialize the database and create tables if needed."""
            try:
                # Create sync engine for table creation
                sync_url = self.db_url.replace('postgresql+asyncpg://', 'postgresql://')
                sync_url = sync_url.replace('sqlite+aiosqlite://', 'sqlite://')
                engine = create_engine(sync_url)
                Base.metadata.create_all(bind=engine)
                engine.dispose()
                
                # Create async engine for operations
                self.engine = create_async_engine(
                    self.db_url,
                    echo=False,
                    pool_pre_ping=True
                )
                self.async_session = sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                
                logger.info(f"Successfully initialized SQLAlchemy database connection to {self.db_url}")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
        
        async def log_event(self, event_type: str, request_id: str, run_id: str = 'unknown', **kwargs) -> None:
            """Log an event to the database."""
            try:
                # Generate timestamp when the function is called
                timestamp = time.time()
                event_data = json.dumps(kwargs, ensure_ascii=False)
                
                async with self.async_session() as session:
                    async with session.begin():
                        session.add(Event(
                            timestamp=timestamp,
                            event_type=event_type,
                            request_id=request_id,
                            run_id=run_id,
                            event_data=event_data
                        ))
                
                logger.info(f"Successfully logged event: {event_type} for request_id: {request_id}, run_id: {run_id}")
            except Exception as e:
                logger.error(f"Failed to log event: {e}")
        
        async def get_events_by_request_id(self, request_id: str, run_id: str = None) -> List[Dict[str, Any]]:
            """Retrieve all events for a specific request ID, optionally filtered by run_id."""
            try:
                async with self.async_session() as session:
                    async with session.begin():
                        if run_id:
                            result = await session.execute(
                                text("SELECT * FROM events WHERE request_id = :request_id AND run_id = :run_id ORDER BY timestamp"),
                                {"request_id": request_id, "run_id": run_id}
                            )
                        else:
                            result = await session.execute(
                                text("SELECT * FROM events WHERE request_id = :request_id ORDER BY timestamp"),
                                {"request_id": request_id}
                            )
                        rows = result.fetchall()
                
                events = []
                for row in rows:
                    event_dict = dict(row._mapping)
                    event_dict['event_data'] = json.loads(event_dict['event_data'])
                    events.append(event_dict)
                
                logger.info(f"Successfully retrieved {len(events)} events for request_id: {request_id}")
                return events
                
            except Exception as e:
                logger.error(f"Failed to retrieve events: {e}")
                return []
        
        async def check_and_log_tool_call(self, normalized_call: str, function_name: str, arguments: Dict[str, Any], 
                                         request_id: str, branch_id: str = "main", run_id: str = "unknown") -> bool:
            """Check if a tool call exists and log it if not. Return True if it existed before."""
            try:
                args_str = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                timestamp = time.time()
                
                async with self.async_session() as session:
                    async with session.begin():
                        # Check if the tool call already exists
                        result = await session.execute(
                            text("SELECT id FROM tool_calls WHERE normalized_call = :normalized_call"),
                            {"normalized_call": normalized_call}
                        )
                        existing = result.fetchone()
                        
                        if existing:
                            # Increment occurrence count
                            await session.execute(
                                text("UPDATE tool_calls SET occurrence_count = occurrence_count + 1 WHERE id = :id"),
                                {"id": existing.id}
                            )
                            logger.info(f"Successfully found and incremented existing tool call: {function_name} for normalized_call: {normalized_call}")
                            return True
                        else:
                            # Insert new tool call
                            session.add(ToolCall(
                                normalized_call=normalized_call,
                                function_name=function_name,
                                arguments=args_str,
                                first_occurrence_time=timestamp,
                                first_occurrence_request_id=request_id,
                                first_occurrence_run_id=run_id
                            ))
                            logger.info(f"Successfully logged new tool call: {function_name} for normalized_call: {normalized_call}, request_id: {request_id}, run_id: {run_id}")
                            return False
                
            except Exception as e:
                logger.error(f"Failed to check and log tool call: {e}")
                return False
        
        async def get_previous_tool_call(self, normalized_call: str) -> Optional[Dict[str, Any]]:
            """Get information about a previous occurrence of a tool call."""
            try:
                async with self.async_session() as session:
                    async with session.begin():
                        result = await session.execute(
                            text("SELECT * FROM tool_calls WHERE normalized_call = :normalized_call"),
                            {"normalized_call": normalized_call}
                        )
                        row = result.fetchone()
                
                if row:
                    tool_call = dict(row._mapping)
                    tool_call['arguments'] = json.loads(tool_call['arguments'])
                    logger.info(f"Successfully retrieved previous tool call: {tool_call['function_name']} for normalized_call: {normalized_call}")
                    return tool_call
                else:
                    logger.info(f"No previous tool call found for normalized_call: {normalized_call}")
                    return None
                
            except Exception as e:
                logger.error(f"Failed to retrieve previous tool call: {e}")
                return None
            
        async def get_events_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
            """Retrieve all events for a specific run_id."""
            try:
                async with self.async_session() as session:
                    async with session.begin():
                        result = await session.execute(
                            text("SELECT * FROM events WHERE run_id = :run_id ORDER BY timestamp"),
                            {"run_id": run_id}
                        )
                        rows = result.fetchall()
                
                events = []
                for row in rows:
                    event_dict = dict(row._mapping)
                    event_dict['event_data'] = json.loads(event_dict['event_data'])
                    events.append(event_dict)
                
                logger.info(f"Successfully retrieved {len(events)} events for run_id: {run_id}")
                return events
                
            except Exception as e:
                logger.error(f"Failed to retrieve events by run_id: {e}")
                return []


@lru_cache(maxsize=1)
def get_event_db(db_type: str = "sqlite", db_path: str = "./events.db", db_url: Optional[str] = None) -> EventDB:
    """Factory function to get the appropriate EventDB instance.
    
    Priority order:
    1. Use db_url if provided
    2. Use DATABASE_URL environment variable if available
    3. Fall back to specified db_type and db_path
    """
    # Check for DATABASE_URL environment variable
    env_db_url = os.environ.get("DATABASE_URL")
    final_db_url = db_url or env_db_url
    
    if final_db_url:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for database URL support")
        return SQLAlchemyEventDB(final_db_url)
    
    if db_type == "sqlite":
        return SQLiteEventDB(db_path)
    elif db_type == "dummy":
        return DummyEventDB()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


# Global instance for easy access
global_event_db = get_event_db()


async def log_event(event_type: str, request_id: str, run_id: str = 'unknown', **kwargs) -> None:
    """Convenience function to log an event using the global event DB instance."""
    await global_event_db.log_event(event_type, request_id, run_id, **kwargs)


async def get_events_by_request_id(request_id: str, run_id: str = None) -> List[Dict[str, Any]]:
    """Convenience function to get events by request ID using the global event DB instance, optionally filtered by run_id."""
    return await global_event_db.get_events_by_request_id(request_id, run_id)


async def get_events_by_run_id(run_id: str) -> List[Dict[str, Any]]:
    """Convenience function to get all events for a specific run_id using the global event DB instance."""
    return await global_event_db.get_events_by_run_id(run_id)
