"""
Streamlit authentication and session management
"""

import streamlit as st
from database.auth import auth
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('session')

def init_session_state():
    """Initialize session state variables if they don't exist"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    
    if 'auth_message' not in st.session_state:
        st.session_state.auth_message = None
    
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = None

def login_user(username: str, password: str) -> None:
    """
    Attempt to log in a user
    
    Args:
        username: Username
        password: Password
    """
    try:
        # Validate inputs
        if not username or not password:
            st.session_state.auth_message = "Please enter both username and password"
            st.session_state.auth_status = "error"
            return
        
        # Attempt login
        user_data = auth.login(username, password)
        
        # Set session variables on success
        st.session_state.user = user_data
        st.session_state.is_authenticated = True
        st.session_state.auth_message = f"Welcome back, {username}!"
        st.session_state.auth_status = "success"
        
        logger.info(f"User logged in: {username}")
        
        # Refresh the page
        st.rerun()
    except Exception as e:
        st.session_state.auth_message = str(e)
        st.session_state.auth_status = "error"
        logger.error(f"Login failed for {username}: {str(e)}")

def register_user(username: str, password: str, password_confirm: str, email: Optional[str] = None) -> None:
    """
    Register a new user
    
    Args:
        username: Username
        password: Password
        password_confirm: Password confirmation
        email: Email (optional)
    """
    try:
        # Validate inputs
        if not username or not password:
            st.session_state.auth_message = "Please enter both username and password"
            st.session_state.auth_status = "error"
            return
        
        if password != password_confirm:
            st.session_state.auth_message = "Passwords do not match"
            st.session_state.auth_status = "error"
            return
        
        # Attempt registration
        user_data = auth.create_user(username, password, email)
        
        # Set session variables on success
        st.session_state.auth_message = f"Registration successful! Welcome, {username}!"
        st.session_state.auth_status = "success"
        
        # Auto-login after registration
        login_user(username, password)
        
        logger.info(f"New user registered: {username}")
        
        # Refresh the page
        st.rerun()
    except Exception as e:
        st.session_state.auth_message = str(e)
        st.session_state.auth_status = "error"
        logger.error(f"Registration failed for {username}: {str(e)}")

def logout_user() -> None:
    """Log out the current user"""
    if st.session_state.user:
        username = st.session_state.user['username']
        logger.info(f"User logged out: {username}")
    
    # Clear session variables
    st.session_state.user = None
    st.session_state.is_authenticated = False
    st.session_state.auth_message = "You have been logged out"
    st.session_state.auth_status = "info"

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get the currently logged in user"""
    return st.session_state.user

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.is_authenticated

def render_login_form():
    """Render the login form"""
    with st.form("login_form"):
        st.subheader("🔑 Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("Login", use_container_width=True)
        
        if login_button:
            login_user(username, password)

def render_register_form():
    """Render the registration form"""
    with st.form("register_form"):
        st.subheader("👤 Create Account")
        username = st.text_input("Username", help="At least 3 characters", key="register_username")
        email = st.text_input("Email (optional)", key="register_email")
        password = st.text_input("Password", type="password", help="At least 6 characters", key="register_password")
        password_confirm = st.text_input("Confirm Password", type="password", key="register_password_confirm")
        
        register_button = st.form_submit_button("Register", use_container_width=True)
        
        if register_button:
            register_user(username, password, password_confirm, email)

def render_logout_button():
    """Render logout button"""
    if st.button("Logout"):
        logout_user()
        st.rerun()

def render_auth_message():
    """Render authentication message if present"""
    if st.session_state.auth_message:
        status = st.session_state.auth_status or "info"
        
        if status == "success":
            st.success(st.session_state.auth_message)
        elif status == "error":
            st.error(st.session_state.auth_message)
        else:
            st.info(st.session_state.auth_message)
        
        # Don't clear message after displaying - it's handled after page refresh
        # We'll keep it for this rendering cycle

def render_auth_ui():
    """Render the complete authentication UI"""
    init_session_state()
    render_auth_message()
    
    if is_authenticated():
        # If user is already logged in, just show their info in the sidebar
        # This is handled in the main app now
        pass
    else:
        # Not authenticated, show login/register forms
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.container():
                st.markdown("""
                <style>
                    .auth-container {
                        background-color: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
                render_login_form()
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            with st.container():
                st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
                render_register_form()
                st.markdown("</div>", unsafe_allow_html=True)
