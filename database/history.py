"""
Search history management module for the water quality prediction system
"""

from typing import Dict, List, Any, Optional
import logging
from database.db_setup import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('history')

class SearchHistoryManager:
    """Search history manager for the water quality application"""
    
    @staticmethod
    def save_search(user_id: Optional[int], water_params: Dict, prediction_result: Dict) -> int:
        """
        Save a search to history
        
        Args:
            user_id: User ID (None for anonymous users)
            water_params: Water quality parameters
            prediction_result: Prediction result data
            
        Returns:
            search_id: ID of the saved search
        """
        try:
            # Extract key prediction data
            is_safe = 1 if prediction_result['safety_status'] == 'SAFE' else 0
            confidence = prediction_result['confidence']
            
            # Save search to database
            search_id = db.save_search(user_id, water_params, is_safe, confidence)
            
            # If detailed report exists, save it too
            if 'detailed_report' in prediction_result:
                report = prediction_result['detailed_report']
                db.save_detailed_report(
                    search_id,
                    report['parameter_analysis'],
                    report['suggestions'],
                    report['next_steps']
                )
            
            logger.info(f"Search saved for user ID: {user_id}, search ID: {search_id}")
            return search_id
        except Exception as e:
            logger.error(f"Error saving search history: {str(e)}")
            raise
    
    @staticmethod
    def get_user_history(user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get search history for a user
        
        Args:
            user_id: User ID
            limit: Maximum number of records to return
            
        Returns:
            List of search history records
        """
        try:
            # Get history from database
            history = db.get_user_search_history(user_id, limit)
            
            # Format for display
            formatted_history = []
            for record in history:
                formatted_record = {
                    'search_id': record['search_id'],
                    'timestamp': record['search_timestamp'],
                    'water_params': record['water_params'],
                    'result': 'SAFE' if record['prediction_result'] == 1 else 'UNSAFE',
                    'confidence': record['confidence']
                }
                formatted_history.append(formatted_record)
            
            logger.info(f"Retrieved {len(formatted_history)} history records for user ID: {user_id}")
            return formatted_history
        except Exception as e:
            logger.error(f"Error retrieving search history: {str(e)}")
            raise
    
    @staticmethod
    def get_search_details(search_id: int) -> Dict:
        """
        Get detailed information for a specific search
        
        Args:
            search_id: Search ID
            
        Returns:
            Detailed search and report data
        """
        try:
            # Get combined search and report data
            details = db.get_search_with_report(search_id)
            
            if not details:
                raise ValueError(f"Search ID {search_id} not found")
            
            # Format for display
            formatted_details = {
                'search_id': details['search_id'],
                'timestamp': details['search_timestamp'],
                'water_params': details['water_params'],
                'result': 'SAFE' if details['prediction_result'] == 1 else 'UNSAFE',
                'confidence': details['confidence']
            }
            
            # Add report data if available
            if 'parameter_analysis' in details and details['parameter_analysis']:
                formatted_details['parameter_analysis'] = details['parameter_analysis']
                formatted_details['suggestions'] = details['suggestions']
                formatted_details['next_steps'] = details['next_steps']
            
            logger.info(f"Retrieved details for search ID: {search_id}")
            return formatted_details
        except Exception as e:
            logger.error(f"Error retrieving search details: {str(e)}")
            raise

# Initialize history manager on module import
history = SearchHistoryManager()
