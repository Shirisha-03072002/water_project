"""
Authentication module for the water quality prediction system
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from database.db_setup import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auth')

# JWT configuration
JWT_SECRET = "water_quality_secret_key"  # In production, this should be stored securely
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class AuthManager:
    """Authentication manager for the water quality application"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        # Generate salt and hash password
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    
    @staticmethod
    def create_user(username: str, password: str, email: Optional[str] = None) -> Dict:
        """
        Create a new user
        
        Args:
            username: User's username
            password: Plain text password
            email: User's email (optional)
            
        Returns:
            User data including user_id
        """
        try:
            # Validate input
            if not username or not password:
                raise ValueError("Username and password are required")
            
            if len(username) < 3:
                raise ValueError("Username must be at least 3 characters")
            
            if len(password) < 6:
                raise ValueError("Password must be at least 6 characters")
            
            # Hash password
            password_hash = AuthManager.hash_password(password)
            
            # Create user in database
            user_id = db.create_user(username, password_hash, email)
            
            return {
                "user_id": user_id,
                "username": username,
                "email": email
            }
        except Exception as e:
            logger.error(f"User creation failed: {str(e)}")
            raise
    
    @staticmethod
    def login(username: str, password: str) -> Dict:
        """
        Authenticate a user and generate JWT token
        
        Args:
            username: User's username
            password: Plain text password
            
        Returns:
            Dict with user data and JWT token if successful
        """
        try:
            # Get user from database
            user = db.get_user_by_username(username)
            
            if not user:
                raise ValueError("Invalid username or password")
            
            # Verify password
            if not AuthManager.verify_password(password, user['password_hash']):
                raise ValueError("Invalid username or password")
            
            # Update last login
            db.update_last_login(user['user_id'])
            
            # Generate token
            token = AuthManager.generate_token(user)
            
            return {
                "user_id": user['user_id'],
                "username": user['username'],
                "email": user['email'],
                "token": token
            }
        except Exception as e:
            logger.error(f"Login failed for {username}: {str(e)}")
            raise
    
    @staticmethod
    def generate_token(user: Dict) -> str:
        """
        Generate JWT token for authenticated user
        
        Args:
            user: User data
            
        Returns:
            JWT token string
        """
        # Set expiration time
        expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        # Create payload
        payload = {
            "sub": user['user_id'],
            "username": user['username'],
            "exp": expiration
        }
        
        # Generate token
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        return token
    
    @staticmethod
    def validate_token(token: str) -> Dict:
        """
        Validate a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            User data if token is valid
        """
        try:
            # Decode token
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Get user from database
            user_id = payload["sub"]
            username = payload["username"]
            
            return {
                "user_id": user_id,
                "username": username,
                "is_authenticated": True
            }
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            logger.error("Invalid token")
            raise ValueError("Invalid token")

# Initialize auth manager on module import
auth = AuthManager()
