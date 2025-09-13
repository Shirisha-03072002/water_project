"""
Database setup and management for the water quality prediction system
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database')

class DatabaseManager:
    """SQLite database manager for the water quality application"""
    
    def __init__(self, db_path: str = "database/water_quality.db"):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Initialize database if it doesn't exist
        self._connect()
        self._create_tables()
        self._disconnect()
    
    def _connect(self) -> None:
        """Create a database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.cursor = self.conn.cursor()
            logger.debug("Database connection established")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _disconnect(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.debug("Database connection closed")
    
    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist"""
        try:
            # Users table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
            ''')
            
            # Search history table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                search_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                water_params TEXT NOT NULL,  -- JSON string of parameters
                prediction_result INTEGER,   -- 1 for safe, 0 for unsafe
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Detailed reports table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detailed_reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER UNIQUE,
                parameter_analysis TEXT,  -- JSON string
                suggestions TEXT,         -- JSON string
                next_steps TEXT,          -- JSON string
                FOREIGN KEY (search_id) REFERENCES search_history (search_id)
            )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    # User Management Functions
    
    def create_user(self, username: str, password_hash: str, email: Optional[str] = None) -> int:
        """
        Create a new user
        
        Args:
            username: User's username
            password_hash: Hashed password
            email: User's email (optional)
            
        Returns:
            user_id: ID of the new user
        """
        try:
            self._connect()
            
            self.cursor.execute(
                "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                (username, password_hash, email)
            )
            
            self.conn.commit()
            user_id = self.cursor.lastrowid
            logger.info(f"User created: {username} (ID: {user_id})")
            
            return user_id
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed - likely duplicate username/email: {e}")
            raise ValueError("Username or email already exists")
        except sqlite3.Error as e:
            logger.error(f"User creation error: {e}")
            raise
        finally:
            self._disconnect()
    
    def get_user_by_username(self, username: str) -> Dict[str, Any]:
        """
        Get user by username
        
        Args:
            username: User's username
            
        Returns:
            User data as dictionary
        """
        try:
            self._connect()
            
            self.cursor.execute(
                "SELECT * FROM users WHERE username = ?", 
                (username,)
            )
            
            user = self.cursor.fetchone()
            
            if user:
                # Convert to dictionary
                user_dict = dict(user)
                logger.debug(f"User found: {username}")
                return user_dict
            else:
                logger.debug(f"User not found: {username}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user: {e}")
            raise
        finally:
            self._disconnect()
    
    def update_last_login(self, user_id: int) -> None:
        """
        Update user's last login timestamp
        
        Args:
            user_id: User ID
        """
        try:
            self._connect()
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.cursor.execute(
                "UPDATE users SET last_login = ? WHERE user_id = ?",
                (current_time, user_id)
            )
            
            self.conn.commit()
            logger.debug(f"Updated last login for user ID: {user_id}")
        except sqlite3.Error as e:
            logger.error(f"Error updating last login: {e}")
            raise
        finally:
            self._disconnect()
    
    # Search History Functions
    
    def save_search(self, user_id: Optional[int], water_params: Dict[str, float],
                   prediction_result: int, confidence: float) -> int:
        """
        Save search to history
        
        Args:
            user_id: User ID (can be None for anonymous users)
            water_params: Water quality parameters
            prediction_result: 1 for safe, 0 for unsafe
            confidence: Prediction confidence
            
        Returns:
            search_id: ID of the saved search
        """
        try:
            self._connect()
            
            # Convert water params to JSON
            params_json = json.dumps(water_params)
            
            self.cursor.execute(
                """
                INSERT INTO search_history 
                (user_id, water_params, prediction_result, confidence) 
                VALUES (?, ?, ?, ?)
                """,
                (user_id, params_json, prediction_result, confidence)
            )
            
            self.conn.commit()
            search_id = self.cursor.lastrowid
            
            logger.info(f"Search saved: ID={search_id}, User ID={user_id}, Result={prediction_result}")
            return search_id
        except sqlite3.Error as e:
            logger.error(f"Error saving search: {e}")
            raise
        finally:
            self._disconnect()
    
    def save_detailed_report(self, search_id: int, parameter_analysis: Dict,
                           suggestions: List, next_steps: List) -> int:
        """
        Save detailed report for a search
        
        Args:
            search_id: Search ID
            parameter_analysis: Parameter analysis results
            suggestions: Suggested treatments
            next_steps: Recommended next steps
            
        Returns:
            report_id: ID of the saved report
        """
        try:
            self._connect()
            
            # Convert to JSON
            param_analysis_json = json.dumps(parameter_analysis)
            suggestions_json = json.dumps(suggestions)
            next_steps_json = json.dumps(next_steps)
            
            self.cursor.execute(
                """
                INSERT INTO detailed_reports
                (search_id, parameter_analysis, suggestions, next_steps)
                VALUES (?, ?, ?, ?)
                """,
                (search_id, param_analysis_json, suggestions_json, next_steps_json)
            )
            
            self.conn.commit()
            report_id = self.cursor.lastrowid
            
            logger.info(f"Detailed report saved: ID={report_id}, Search ID={search_id}")
            return report_id
        except sqlite3.Error as e:
            logger.error(f"Error saving detailed report: {e}")
            raise
        finally:
            self._disconnect()
    
    def get_user_search_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get search history for a user
        
        Args:
            user_id: User ID
            limit: Maximum number of records to return
            
        Returns:
            List of search history records
        """
        try:
            self._connect()
            
            self.cursor.execute(
                """
                SELECT * FROM search_history
                WHERE user_id = ?
                ORDER BY search_timestamp DESC
                LIMIT ?
                """,
                (user_id, limit)
            )
            
            history = self.cursor.fetchall()
            
            # Convert to list of dictionaries and parse JSON
            result = []
            for record in history:
                record_dict = dict(record)
                # Ensure water_params is a string before parsing
                if isinstance(record_dict['water_params'], bytes):
                    record_dict['water_params'] = record_dict['water_params'].decode('utf-8')
                record_dict['water_params'] = json.loads(record_dict['water_params'])
                
                # Ensure timestamp is a string
                if isinstance(record_dict['search_timestamp'], bytes):
                    record_dict['search_timestamp'] = record_dict['search_timestamp'].decode('utf-8')
                
                result.append(record_dict)
            
            logger.debug(f"Retrieved {len(result)} search history records for user ID: {user_id}")
            return result
        except sqlite3.Error as e:
            logger.error(f"Error retrieving search history: {e}")
            raise
        finally:
            self._disconnect()
    
    def get_detailed_report(self, search_id: int) -> Dict:
        """
        Get detailed report for a search
        
        Args:
            search_id: Search ID
            
        Returns:
            Detailed report as dictionary
        """
        try:
            self._connect()
            
            self.cursor.execute(
                "SELECT * FROM detailed_reports WHERE search_id = ?",
                (search_id,)
            )
            
            report = self.cursor.fetchone()
            
            if report:
                # Convert to dictionary and parse JSON
                report_dict = dict(report)
                
                # Handle bytes in parameter_analysis
                if isinstance(report_dict['parameter_analysis'], bytes):
                    report_dict['parameter_analysis'] = report_dict['parameter_analysis'].decode('utf-8')
                report_dict['parameter_analysis'] = json.loads(report_dict['parameter_analysis'])
                
                # Handle bytes in suggestions
                if isinstance(report_dict['suggestions'], bytes):
                    report_dict['suggestions'] = report_dict['suggestions'].decode('utf-8')
                report_dict['suggestions'] = json.loads(report_dict['suggestions'])
                
                # Handle bytes in next_steps
                if isinstance(report_dict['next_steps'], bytes):
                    report_dict['next_steps'] = report_dict['next_steps'].decode('utf-8')
                report_dict['next_steps'] = json.loads(report_dict['next_steps'])
                
                logger.debug(f"Retrieved detailed report for search ID: {search_id}")
                return report_dict
            else:
                logger.debug(f"No detailed report found for search ID: {search_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving detailed report: {e}")
            raise
        finally:
            self._disconnect()
    
    def get_search_with_report(self, search_id: int) -> Dict:
        """
        Get search with its detailed report
        
        Args:
            search_id: Search ID
            
        Returns:
            Combined search and report data
        """
        try:
            self._connect()
            
            self.cursor.execute(
                """
                SELECT 
                    s.search_id, 
                    s.user_id, 
                    s.search_timestamp,
                    s.water_params,
                    s.prediction_result,
                    s.confidence,
                    d.parameter_analysis,
                    d.suggestions,
                    d.next_steps
                FROM search_history s
                LEFT JOIN detailed_reports d ON s.search_id = d.search_id
                WHERE s.search_id = ?
                """,
                (search_id,)
            )
            
            record = self.cursor.fetchone()
            
            if record:
                # Convert to dictionary and parse JSON
                result = dict(record)
                
                # Handle bytes in water_params
                if isinstance(result['water_params'], bytes):
                    result['water_params'] = result['water_params'].decode('utf-8')
                result['water_params'] = json.loads(result['water_params'])
                
                # Handle bytes in timestamp
                if isinstance(result['search_timestamp'], bytes):
                    result['search_timestamp'] = result['search_timestamp'].decode('utf-8')
                
                # Handle bytes in report data if present
                if result['parameter_analysis']:
                    if isinstance(result['parameter_analysis'], bytes):
                        result['parameter_analysis'] = result['parameter_analysis'].decode('utf-8')
                    result['parameter_analysis'] = json.loads(result['parameter_analysis'])
                    
                    if isinstance(result['suggestions'], bytes):
                        result['suggestions'] = result['suggestions'].decode('utf-8')
                    result['suggestions'] = json.loads(result['suggestions'])
                    
                    if isinstance(result['next_steps'], bytes):
                        result['next_steps'] = result['next_steps'].decode('utf-8')
                    result['next_steps'] = json.loads(result['next_steps'])
                
                logger.debug(f"Retrieved search with report for search ID: {search_id}")
                return result
            else:
                logger.debug(f"No search found for search ID: {search_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving search with report: {e}")
            raise
        finally:
            self._disconnect()

# Initialize database on module import
db = DatabaseManager()
