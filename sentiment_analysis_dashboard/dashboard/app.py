"""
Sentiment Analysis Dashboard - Main Streamlit Application
Enterprise-grade admin dashboard with authentication and analytics.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import os

# Page configuration
st.set_page_config(
    page_title="CipherGuard Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000/api/v1")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .sidebar-info {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    
    .error-msg {
        background: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 10px 0;
    }
    
    .info-msg {
        background: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class APIClient:
    """API client for backend communication."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def set_auth_token(self, token: str):
        """Set authentication token for requests."""
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def clear_auth_token(self):
        """Clear authentication token."""
        if "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and get access token."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current user information."""
        try:
            response = self.session.get(f"{self.base_url}/auth/me")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_comment(self, comment_text: str) -> Dict[str, Any]:
        """Analyze sentiment of a comment."""
        try:
            response = self.session.post(
                f"{self.base_url}/comments/analyze",
                json={"comment_text": comment_text, "store_result": True}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_comments(self, page: int = 1, page_size: int = 10, **filters) -> Dict[str, Any]:
        """Get paginated comments with filters."""
        try:
            params = {"page": page, "page_size": page_size, **filters}
            response = self.session.get(f"{self.base_url}/comments/", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_sentiment_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get sentiment analytics."""
        try:
            response = self.session.get(
                f"{self.base_url}/comments/analytics/sentiment",
                params={"days": days}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_timeseries_analytics(self, days: int = 7, interval_hours: int = 24) -> Dict[str, Any]:
        """Get time-series analytics."""
        try:
            response = self.session.get(
                f"{self.base_url}/comments/analytics/timeseries",
                params={"days": days, "interval_hours": interval_hours}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_recent_negative_comments(self, limit: int = 10, hours: int = 24) -> Dict[str, Any]:
        """Get recent negative comments."""
        try:
            response = self.session.get(
                f"{self.base_url}/comments/recent/negative",
                params={"limit": limit, "hours": hours}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ======================= FRAUD DETECTION API METHODS =======================
    
    def detect_fraud(self, amount: float, merchant: str, device: str, country: str,
                      customer_name: str = None, customer_email: str = None, 
                      customer_phone: str = None, send_alert: bool = True) -> Dict[str, Any]:
        """Detect fraud for a transaction and optionally send alerts."""
        try:
            payload = {
                "amount": amount,
                "merchant": merchant,
                "device": device,
                "country": country,
                "send_alert": send_alert
            }
            
            # Add customer info if provided
            if customer_name:
                payload["customer_name"] = customer_name
            if customer_email:
                payload["customer_email"] = customer_email
            if customer_phone:
                payload["customer_phone"] = customer_phone
            
            response = self.session.post(
                f"{self.base_url}/fraud/detect",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_fraud_transactions(self, page: int = 1, page_size: int = 10, **filters) -> Dict[str, Any]:
        """Get paginated fraud transactions."""
        try:
            params = {"page": page, "page_size": page_size, **filters}
            response = self.session.get(f"{self.base_url}/fraud/transactions", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_fraud_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get fraud analytics."""
        try:
            response = self.session.get(
                f"{self.base_url}/fraud/analytics",
                params={"days": days}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_recent_frauds(self, limit: int = 10, hours: int = 24) -> Dict[str, Any]:
        """Get recent fraudulent transactions."""
        try:
            response = self.session.get(
                f"{self.base_url}/fraud/recent",
                params={"limit": limit, "hours": hours}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_alert_history(self, limit: int = 50, days: int = 30) -> Dict[str, Any]:
        """Get customer alert history."""
        try:
            response = self.session.get(
                f"{self.base_url}/fraud/alerts",
                params={"limit": limit, "days": days}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_alert_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            response = self.session.get(
                f"{self.base_url}/fraud/alerts/stats",
                params={"days": days}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get cached API client instance."""
    return APIClient(API_BASE_URL)

def initialize_session_state():
    """Initialize session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None

def show_login_form():
    """Display login form."""
    st.markdown("""
    <div class="main-header">
        <h1>�️ CipherGuard Analytics Dashboard</h1>
        <h3>Sentiment Analysis & Fraud Detection</h3>
        <p>Please login to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    api_client = get_api_client()
                    
                    with st.spinner("Logging in..."):
                        result = api_client.login(username, password)
                    
                    if result.get("success"):
                        # Set authentication
                        token = result["data"]["access_token"]
                        st.session_state.authenticated = True
                        st.session_state.access_token = token
                        api_client.set_auth_token(token)
                        
                        # Get user info
                        user_result = api_client.get_current_user()
                        if user_result.get("success"):
                            st.session_state.user_info = user_result["data"]
                        
                        st.success("Login successful!")
                        st.experimental_rerun()
                    else:
                        st.error(f"Login failed: {result.get('error', 'Unknown error')}")
                else:
                    st.error("Please enter both username and password.")
        
        st.markdown("""
        <div class="info-msg">
            <strong>Demo Credentials:</strong><br>
            Username: <code>admin</code><br>
            Password: <code>admin123</code>
        </div>
        """, unsafe_allow_html=True)

def show_main_dashboard():
    """Display main dashboard interface."""
    api_client = get_api_client()
    
    # Ensure API client has auth token
    if st.session_state.access_token:
        api_client.set_auth_token(st.session_state.access_token)
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>�️ CipherGuard Analytics Dashboard</h1>
        <p>Sentiment Analysis & Fraud Detection | Welcome, {st.session_state.user_info.get('full_name', 'User')}!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        selected_tab = st.selectbox(
            "Choose a section:",
            ["🏠 Overview", "🔍 Analyze Comments", "�️ Fraud Detection", "�📈 Analytics", "📋 Comment History", "⚙️ System Status"]
        )
        
        # User info
        st.markdown("---")
        st.markdown("### User Info")
        user_info = st.session_state.user_info
        if user_info:
            st.markdown(f"""
            <div class="sidebar-info">
                <strong>Name:</strong> {user_info.get('full_name', 'N/A')}<br>
                <strong>Username:</strong> {user_info.get('username', 'N/A')}<br>
                <strong>Email:</strong> {user_info.get('email', 'N/A')}<br>
                <strong>Role:</strong> {'Admin' if user_info.get('is_admin') else 'User'}
            </div>
            """, unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.access_token = None
            api_client.clear_auth_token()
            st.experimental_rerun()
    
    # Main content based on selected tab
    if selected_tab == "🏠 Overview":
        show_overview_tab(api_client)
    elif selected_tab == "🔍 Analyze Comments":
        show_analyze_tab(api_client)
    elif selected_tab == "�️ Fraud Detection":
        show_fraud_detection_tab(api_client)
    elif selected_tab == "�📈 Analytics":
        show_analytics_tab(api_client)
    elif selected_tab == "📋 Comment History":
        show_history_tab(api_client)
    elif selected_tab == "⚙️ System Status":
        show_system_status_tab(api_client)

def show_overview_tab(api_client: APIClient):
    """Display overview dashboard."""
    st.header("📊 Dashboard Overview")
    
    # Get analytics data
    with st.spinner("Loading dashboard data..."):
        analytics_result = api_client.get_sentiment_analytics(days=30)
        negative_comments_result = api_client.get_recent_negative_comments(limit=5, hours=24)
    
    if analytics_result.get("success"):
        analytics = analytics_result["data"]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments (30d)", f"{analytics['total_comments']:,}")
        
        with col2:
            st.metric("Positive Comments", f"{analytics['positive_count']:,}", 
                     f"{analytics['positive_percentage']:.1f}%")
        
        with col3:
            st.metric("Negative Comments", f"{analytics['negative_count']:,}", 
                     f"{analytics['negative_percentage']:.1f}%")
        
        with col4:
            st.metric("Avg Confidence", f"{analytics['avg_confidence_score']:.3f}")
        
        # Sentiment distribution chart
        st.subheader("📈 Sentiment Distribution (Last 30 Days)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pie chart
            labels = ['Positive', 'Negative', 'Neutral']
            values = [analytics['positive_count'], analytics['negative_count'], analytics['neutral_count']]
            colors = ['#28a745', '#dc3545', '#ffc107']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
            fig.update_traces(marker=dict(colors=colors))
            fig.update_layout(title="Sentiment Distribution", height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics table
            st.markdown("### Statistics")
            stats_df = pd.DataFrame({
                'Sentiment': labels,
                'Count': values,
                'Percentage': [analytics['positive_percentage'], 
                              analytics['negative_percentage'], 
                              analytics['neutral_percentage']]
            })
            st.dataframe(stats_df, hide_index=True)
    
    # Recent negative comments alert
    if negative_comments_result.get("success"):
        negative_comments = negative_comments_result["data"]
        
        if negative_comments:
            st.subheader("🚨 Recent Negative Comments (Last 24h)")
            
            for comment in negative_comments:
                with st.expander(f"Comment {comment['id']} - Confidence: {comment['confidence_score']:.2f}"):
                    st.write(f"**Text:** {comment['comment_text']}")
                    st.write(f"**Time:** {comment['created_at']}")
                    st.write(f"**User:** {comment['user_id']}")

def show_analyze_tab(api_client: APIClient):
    """Display comment analysis interface."""
    st.header("🔍 Analyze Comment Sentiment")
    
    st.markdown("""
    Enter a comment below to analyze its sentiment. The system will classify it as 
    positive, negative, or neutral and provide a confidence score.
    """)
    
    # Comment input form
    with st.form("analyze_form"):
        comment_text = st.text_area(
            "Enter comment to analyze:",
            height=150,
            placeholder="Type your comment here...",
            max_chars=5000
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.form_submit_button("🔍 Analyze Sentiment")
        
        if analyze_button and comment_text.strip():
            with st.spinner("Analyzing sentiment..."):
                result = api_client.analyze_comment(comment_text.strip())
            
            if "sentiment" in result:  # Direct response from analyze endpoint
                sentiment = result["sentiment"]
                confidence = result["confidence_score"]
                processing_time = result.get("processing_time_ms", 0)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Sentiment with color coding
                    if sentiment == "positive":
                        st.success(f"**Sentiment:** {sentiment.title()}")
                    elif sentiment == "negative":
                        st.error(f"**Sentiment:** {sentiment.title()}")
                    else:
                        st.warning(f"**Sentiment:** {sentiment.title()}")
                
                with col2:
                    st.info(f"**Confidence:** {confidence:.3f}")
                
                with col3:
                    st.metric("Processing Time", f"{processing_time:.1f} ms")
                
                # Confidence gauge
                st.subheader("Confidence Score")
                
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Level"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                gauge_fig.update_layout(height=300)
                
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Success message
                st.markdown(f"""
                <div class="success-msg">
                    ✅ Comment analyzed successfully and saved to database with ID: {result.get('id', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        elif analyze_button:
            st.warning("Please enter a comment to analyze.")

def show_fraud_detection_tab(api_client: APIClient):
    """Display fraud detection interface."""
    st.header("🛡️ Fraud Detection")
    
    st.markdown("""
    Analyze transactions for potential fraud using our advanced ML-based detection system.
    The system uses Isolation Forest algorithm combined with rule-based analysis.
    **NEW: Customer alerts are sent via SMS/Email when fraud is detected!**
    """)
    
    # Create two columns for detection and analytics
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Transaction", "📊 Fraud Analytics", "📋 Transaction History", "🔔 Alert History"])
    
    with tab1:
        show_fraud_analyze_section(api_client)
    
    with tab2:
        show_fraud_analytics_section(api_client)
    
    with tab3:
        show_fraud_history_section(api_client)
    
    with tab4:
        show_alert_history_section(api_client)

def show_fraud_analyze_section(api_client: APIClient):
    """Display fraud analysis form."""
    st.subheader("🔍 Analyze New Transaction")
    
    with st.form("fraud_detection_form"):
        st.markdown("#### Transaction Details")
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=1000000.0,
                value=100.0,
                step=10.0,
                help="Enter the transaction amount"
            )
            
            merchant = st.text_input(
                "Merchant Name",
                value="Amazon",
                max_chars=200,
                help="Enter the merchant name"
            )
        
        with col2:
            device = st.selectbox(
                "Device Type",
                options=["desktop", "mobile", "tablet"],
                index=0,
                help="Select the device used for transaction"
            )
            
            country = st.selectbox(
                "Country",
                options=["US", "UK", "CA", "AU", "DE", "FR", "CN", "RU", "BR", "Other"],
                index=0,
                help="Select the country of transaction"
            )
        
        # Customer info for alerts
        st.markdown("---")
        st.markdown("#### 🔔 Customer Alert Settings (Optional)")
        st.info("Provide customer contact info to send real-time fraud alerts via SMS/Email")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customer_name = st.text_input(
                "Customer Name",
                placeholder="John Doe",
                help="Customer's full name"
            )
        
        with col2:
            customer_email = st.text_input(
                "Customer Email",
                placeholder="john@example.com",
                help="Email for fraud alerts"
            )
        
        with col3:
            customer_phone = st.text_input(
                "Customer Phone",
                placeholder="+1234567890",
                help="Phone number for SMS alerts"
            )
        
        send_alert = st.checkbox("📲 Send alert if fraud detected", value=True)
        
        analyze_button = st.form_submit_button("🔍 Analyze for Fraud", use_container_width=True)
        
        if analyze_button:
            with st.spinner("Analyzing transaction for fraud..."):
                result = api_client.detect_fraud(
                    amount=amount, 
                    merchant=merchant, 
                    device=device, 
                    country=country,
                    customer_name=customer_name if customer_name else None,
                    customer_email=customer_email if customer_email else None,
                    customer_phone=customer_phone if customer_phone else None,
                    send_alert=send_alert
                )
            
            if result.get("success"):
                data = result["data"]
                
                # Display results with color coding
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = data.get("risk_level", "UNKNOWN")
                    risk_colors = {
                        "CRITICAL": "🔴",
                        "HIGH": "🟠",
                        "MEDIUM": "🟡",
                        "LOW": "🟢",
                        "VERY_LOW": "🟢"
                    }
                    risk_emoji = risk_colors.get(risk_level, "⚪")
                    
                    if risk_level in ["CRITICAL", "HIGH"]:
                        st.error(f"**Risk Level:** {risk_emoji} {risk_level}")
                    elif risk_level == "MEDIUM":
                        st.warning(f"**Risk Level:** {risk_emoji} {risk_level}")
                    else:
                        st.success(f"**Risk Level:** {risk_emoji} {risk_level}")
                
                with col2:
                    fraud_score = data.get("fraud_score", 0)
                    st.metric("Fraud Score", f"{fraud_score:.3f}")
                
                with col3:
                    is_fraud = data.get("is_fraud", False)
                    if is_fraud:
                        st.error("**Status:** ⚠️ FLAGGED AS FRAUD")
                    else:
                        st.success("**Status:** ✅ LEGITIMATE")
                
                # Fraud score gauge
                st.subheader("Fraud Risk Gauge")
                
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=fraud_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkred" if fraud_score > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "#90EE90"},
                            {'range': [0.3, 0.5], 'color': "#FFFF00"},
                            {'range': [0.5, 0.7], 'color': "#FFA500"},
                            {'range': [0.7, 0.9], 'color': "#FF6347"},
                            {'range': [0.9, 1], 'color': "#DC143C"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                gauge_fig.update_layout(height=300)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Transaction details
                st.markdown("### Transaction Details")
                details_df = pd.DataFrame({
                    'Field': ['Transaction ID', 'Amount', 'Merchant', 'Device', 'Country', 'Timestamp'],
                    'Value': [
                        data.get('transaction_id', 'N/A'),
                        f"${data.get('amount', 0):,.2f}",
                        data.get('merchant', 'N/A'),
                        data.get('device', 'N/A'),
                        data.get('country', 'N/A'),
                        data.get('created_at', 'N/A')
                    ]
                })
                st.dataframe(details_df, hide_index=True, use_container_width=True)
                
                # Success message
                st.markdown(f"""
                <div class="success-msg">
                    ✅ Transaction analyzed and saved with ID: {data.get('transaction_id', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
                
                # Alert status display
                if data.get('alert_sent'):
                    alert_channels = data.get('alert_channels', [])
                    channels_text = ', '.join(alert_channels) if alert_channels else 'N/A'
                    st.markdown("---")
                    st.subheader("🔔 Customer Alert Status")
                    
                    alert_col1, alert_col2 = st.columns(2)
                    with alert_col1:
                        st.success("**Alert Sent:** ✅ Yes")
                        st.info(f"**Channels:** {channels_text}")
                    with alert_col2:
                        if customer_email:
                            st.write(f"📧 Email sent to: {customer_email}")
                        if customer_phone:
                            st.write(f"📱 SMS sent to: {customer_phone}")
                    
                    st.info("⚠️ Customer has been notified about this suspicious transaction.")
                elif send_alert and is_fraud:
                    st.warning("⚠️ Alert requested but no customer contact information provided.")
            
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

def show_fraud_analytics_section(api_client: APIClient):
    """Display fraud analytics dashboard."""
    st.subheader("📊 Fraud Analytics Dashboard")
    
    # Time period selector
    days = st.selectbox(
        "Select time period:",
        options=[7, 30, 90],
        format_func=lambda x: f"Last {x} days",
        index=1
    )
    
    with st.spinner("Loading fraud analytics..."):
        analytics_result = api_client.get_fraud_analytics(days=days)
    
    if analytics_result.get("success"):
        analytics = analytics_result["data"]
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{analytics.get('total_transactions', 0):,}")
        
        with col2:
            fraud_count = analytics.get('fraud_count', 0)
            st.metric("Fraud Detected", f"{fraud_count:,}", 
                     delta=f"{analytics.get('fraud_rate', 0):.1f}%",
                     delta_color="inverse")
        
        with col3:
            st.metric("Legitimate", f"{analytics.get('legitimate_count', 0):,}")
        
        with col4:
            st.metric("Avg Fraud Score", f"{analytics.get('average_fraud_score', 0):.3f}")
        
        # Amount statistics
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Transaction Amount", f"${analytics.get('total_amount', 0):,.2f}")
        
        with col2:
            st.metric("Fraud Amount (at risk)", f"${analytics.get('fraud_amount', 0):,.2f}")
        
        # Charts
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud vs Legitimate pie chart
            st.subheader("🎯 Fraud Distribution")
            
            fraud_count = analytics.get('fraud_count', 0)
            legitimate_count = analytics.get('legitimate_count', 0)
            
            if fraud_count + legitimate_count > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Fraud', 'Legitimate'],
                    values=[fraud_count, legitimate_count],
                    hole=0.4,
                    marker_colors=['#dc3545', '#28a745']
                )])
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No transaction data available")
        
        with col2:
            # Risk level distribution
            st.subheader("📊 Risk Level Distribution")
            
            risk_dist = analytics.get('risk_distribution', {})
            if any(risk_dist.values()):
                risk_df = pd.DataFrame({
                    'Risk Level': list(risk_dist.keys()),
                    'Count': list(risk_dist.values())
                })
                
                colors = {
                    'CRITICAL': '#dc3545',
                    'HIGH': '#fd7e14',
                    'MEDIUM': '#ffc107',
                    'LOW': '#28a745',
                    'VERY_LOW': '#20c997'
                }
                
                fig = px.bar(
                    risk_df,
                    x='Risk Level',
                    y='Count',
                    color='Risk Level',
                    color_discrete_map=colors
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk distribution data available")
    
    else:
        st.error(f"Failed to load analytics: {analytics_result.get('error', 'Unknown error')}")

def show_fraud_history_section(api_client: APIClient):
    """Display fraud transaction history."""
    st.subheader("📋 Transaction History")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_fraud = st.selectbox(
            "Filter by status:",
            options=[None, True, False],
            format_func=lambda x: "All" if x is None else ("Fraud Only" if x else "Legitimate Only"),
            index=0
        )
    
    with col2:
        filter_risk = st.selectbox(
            "Filter by risk level:",
            options=[None, "CRITICAL", "HIGH", "MEDIUM", "LOW", "VERY_LOW"],
            format_func=lambda x: "All" if x is None else x,
            index=0
        )
    
    with col3:
        min_amount = st.number_input("Min Amount ($)", min_value=0.0, value=0.0, step=100.0)
    
    with col4:
        max_amount = st.number_input("Max Amount ($)", min_value=0.0, value=0.0, step=100.0)
    
    # Pagination
    page = st.number_input("Page", min_value=1, value=1, step=1)
    page_size = st.selectbox("Results per page:", options=[10, 25, 50], index=0)
    
    # Build filters
    filters = {"days": 90}
    if filter_fraud is not None:
        filters["is_fraud"] = filter_fraud
    if filter_risk:
        filters["risk_level"] = filter_risk
    if min_amount > 0:
        filters["min_amount"] = min_amount
    if max_amount > 0:
        filters["max_amount"] = max_amount
    
    with st.spinner("Loading transactions..."):
        result = api_client.get_fraud_transactions(page=page, page_size=page_size, **filters)
    
    if result.get("success"):
        data = result["data"]
        transactions = data.get("transactions", [])
        
        st.markdown(f"**Total Results:** {data.get('total_count', 0)} | **Page:** {data.get('page', 1)} of {data.get('total_pages', 1)}")
        
        if transactions:
            # Create DataFrame for display
            df = pd.DataFrame([
                {
                    'Transaction ID': t['transaction_id'],
                    'Amount': f"${t['amount']:,.2f}",
                    'Merchant': t['merchant'],
                    'Device': t['device'],
                    'Country': t['country'],
                    'Fraud': '⚠️ Yes' if t['is_fraud'] else '✅ No',
                    'Score': f"{t['fraud_score']:.3f}",
                    'Risk': t['risk_level'],
                    'Time': t['created_at'][:19] if t.get('created_at') else 'N/A'
                }
                for t in transactions
            ])
            
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No transactions found matching the criteria.")
    
    else:
        st.error(f"Failed to load transactions: {result.get('error', 'Unknown error')}")

def show_alert_history_section(api_client: APIClient):
    """Display customer alert history and statistics."""
    st.subheader("🔔 Customer Alert History")
    
    # Alert Statistics
    st.markdown("### Alert Statistics")
    
    with st.spinner("Loading alert statistics..."):
        stats_result = api_client.get_alert_stats()
    
    if stats_result.get("success"):
        stats = stats_result["data"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", f"{stats.get('total_alerts', 0):,}")
        
        with col2:
            st.metric("Emails Sent", f"{stats.get('email_sent', 0):,}")
        
        with col3:
            st.metric("SMS Sent", f"{stats.get('sms_sent', 0):,}")
        
        with col4:
            success_rate = stats.get('success_rate', 0)
            st.metric("Success Rate", f"{success_rate:.1f}%")
    else:
        st.warning("Could not load alert statistics")
    
    st.markdown("---")
    
    # Alert History
    st.markdown("### Recent Alerts")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_filter = st.selectbox(
            "Time period:",
            options=[7, 30, 90],
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        alert_type_filter = st.selectbox(
            "Alert Type:",
            options=[None, "suspicious_transaction", "high_amount", "blocked_transaction"],
            format_func=lambda x: "All" if x is None else x.replace("_", " ").title()
        )
    
    with col3:
        limit = st.selectbox("Results:", options=[20, 50, 100], index=1, key="alert_limit")
    
    with st.spinner("Loading alerts..."):
        alerts_result = api_client.get_alert_history(limit=limit, days=days_filter)
    
    if alerts_result.get("success"):
        data = alerts_result["data"]
        alerts = data.get("alerts", [])
        
        st.markdown(f"**Total Alerts:** {data.get('total_count', len(alerts))}")
        
        if alerts:
            for alert in alerts:
                with st.expander(
                    f"🔔 Alert {alert.get('alert_id', 'N/A')[:8]}... - {alert.get('alert_type', 'N/A').replace('_', ' ').title()}",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Transaction ID:** `{alert.get('transaction_id', 'N/A')}`")
                        st.markdown(f"**Customer:** {alert.get('customer_name', 'N/A')}")
                        st.markdown(f"**Email:** {alert.get('customer_email', 'N/A')}")
                        st.markdown(f"**Phone:** {alert.get('customer_phone', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Alert Type:** {alert.get('alert_type', 'N/A').replace('_', ' ').title()}")
                        st.markdown(f"**Channels:** {', '.join(alert.get('channels_used', []))}")
                        
                        # Delivery status
                        if alert.get('email_delivered'):
                            st.success("📧 Email delivered")
                        if alert.get('sms_delivered'):
                            st.success("📱 SMS delivered")
                        if alert.get('push_delivered'):
                            st.success("🔔 Push delivered")
                    
                    st.markdown(f"**Sent at:** {alert.get('created_at', 'N/A')[:19] if alert.get('created_at') else 'N/A'}")
        else:
            st.info("No alerts found. Alerts are sent when fraud is detected and customer contact info is provided.")
    else:
        st.error(f"Failed to load alerts: {alerts_result.get('error', 'Unknown error')}")
    
    # Info box
    st.markdown("---")
    st.info("""
    **How Customer Alerts Work:**
    
    1. When analyzing a transaction, provide customer email/phone
    2. If fraud is detected, an alert is automatically sent
    3. Customers receive notifications via Email, SMS, or Push
    4. All alerts are logged here for tracking
    """)

def show_analytics_tab(api_client: APIClient):
    """Display analytics dashboard."""
    st.header("📈 Sentiment Analytics")
    
    # Time period selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        time_period = st.selectbox(
            "Select time period:",
            ["Last 7 days", "Last 30 days", "Last 90 days"],
            index=1
        )
    
    period_days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
    days = period_days[time_period]
    
    # Load analytics data
    with st.spinner("Loading analytics..."):
        analytics_result = api_client.get_sentiment_analytics(days=days)
        timeseries_result = api_client.get_timeseries_analytics(days=min(days, 30))
    
    if analytics_result.get("success"):
        analytics = analytics_result["data"]
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments", f"{analytics['total_comments']:,}")
        
        with col2:
            st.metric("Positive Rate", f"{analytics['positive_percentage']:.1f}%")
        
        with col3:
            st.metric("Negative Rate", f"{analytics['negative_percentage']:.1f}%")
        
        with col4:
            st.metric("Avg Confidence", f"{analytics['avg_confidence_score']:.3f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            sentiments = ['Positive', 'Negative', 'Neutral']
            counts = [analytics['positive_count'], analytics['negative_count'], analytics['neutral_count']]
            colors = ['#28a745', '#dc3545', '#ffc107']
            
            bar_fig = px.bar(
                x=sentiments, 
                y=counts, 
                color=sentiments,
                color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107'},
                title="Sentiment Counts"
            )
            bar_fig.update_layout(showlegend=False)
            
            st.plotly_chart(bar_fig, use_container_width=True)
        
        with col2:
            # Donut chart
            donut_fig = go.Figure(data=[go.Pie(
                labels=sentiments, 
                values=counts, 
                hole=0.5,
                marker_colors=colors
            )])
            donut_fig.update_layout(title="Sentiment Distribution", height=400)
            
            st.plotly_chart(donut_fig, use_container_width=True)
    
    # Time series chart
    if timeseries_result.get("success"):
        st.subheader("📊 Sentiment Trends Over Time")
        
        timeseries = timeseries_result["data"]
        data_points = timeseries["data_points"]
        
        if data_points:
            df = pd.DataFrame(data_points)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Line chart
            line_fig = go.Figure()
            
            line_fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['positive_count'],
                mode='lines+markers',
                name='Positive',
                line=dict(color='#28a745')
            ))
            
            line_fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['negative_count'],
                mode='lines+markers',
                name='Negative',
                line=dict(color='#dc3545')
            ))
            
            line_fig.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['neutral_count'],
                mode='lines+markers',
                name='Neutral',
                line=dict(color='#ffc107')
            ))
            
            line_fig.update_layout(
                title="Sentiment Trends",
                xaxis_title="Time",
                yaxis_title="Number of Comments",
                height=400
            )
            
            st.plotly_chart(line_fig, use_container_width=True)

def show_history_tab(api_client: APIClient):
    """Display comment history with filters."""
    st.header("📋 Comment History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All", "positive", "negative", "neutral"]
        )
    
    with col2:
        search_text = st.text_input("Search in comments:")
    
    with col3:
        page_size = st.selectbox("Comments per page:", [10, 25, 50, 100], index=1)
    
    # Pagination
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    # Load comments
    filters = {}
    if sentiment_filter != "All":
        filters["sentiment"] = sentiment_filter
    if search_text:
        filters["search"] = search_text
    
    with st.spinner("Loading comments..."):
        result = api_client.get_comments(
            page=st.session_state.current_page, 
            page_size=page_size, 
            **filters
        )
    
    if "items" in result:  # Direct response from comments endpoint
        comments = result["items"]
        total = result["total"]
        total_pages = result["total_pages"]
        current_page = result["page"]
        
        # Display total count
        st.write(f"**Total comments:** {total:,}")
        
        if comments:
            # Comments table
            for i, comment in enumerate(comments):
                with st.expander(f"Comment {comment['id']} - {comment['sentiment'].title()} ({comment['confidence_score']:.2f})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Text:** {comment['comment_text']}")
                    
                    with col2:
                        st.write(f"**Sentiment:** {comment['sentiment'].title()}")
                        st.write(f"**Confidence:** {comment['confidence_score']:.3f}")
                        st.write(f"**Time:** {comment['created_at'][:19]}")
                        st.write(f"**User:** {comment['user_id']}")
                        st.write(f"**Processing:** {comment['processing_time_ms']:.1f}ms")
                
            # Pagination controls
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if current_page > 1:
                    if st.button("← Previous"):
                        st.session_state.current_page = current_page - 1
                        st.experimental_rerun()
            
            with col2:
                st.write(f"Page {current_page} of {total_pages}")
            
            with col3:
                if current_page < total_pages:
                    if st.button("Next →"):
                        st.session_state.current_page = current_page + 1
                        st.experimental_rerun()
        
        else:
            st.info("No comments found matching the criteria.")
    
    else:
        st.error(f"Failed to load comments: {result.get('error', 'Unknown error')}")

def show_system_status_tab(api_client: APIClient):
    """Display system status and health information."""
    st.header("⚙️ System Status")
    
    # Health check
    with st.spinner("Checking system health..."):
        try:
            health_response = requests.get(f"{API_BASE_URL}/health/")
            detailed_health_response = requests.get(
                f"{API_BASE_URL}/health/detailed",
                headers={"Authorization": f"Bearer {st.session_state.access_token}"}
            )
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            return
    
    # Basic health
    if health_response.status_code == 200:
        health_data = health_response.json()["data"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("🟢 API Status: Healthy")
        
        with col2:
            st.info(f"⏱️ Uptime: {health_data['uptime_seconds']:.0f}s")
        
        with col3:
            st.info(f"🏷️ Version: {health_data['version']}")
    
    else:
        st.error("🔴 API Status: Unhealthy")
    
    # Detailed health
    if detailed_health_response.status_code == 200:
        detailed_data = detailed_health_response.json()["data"]
        
        # Application health
        st.subheader("📱 Application Health")
        app_data = detailed_data.get("application", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Status:** {app_data.get('status', 'Unknown')}")
            st.write(f"**Uptime:** {app_data.get('uptime_seconds', 0):.0f} seconds")
        with col2:
            st.write(f"**Version:** {app_data.get('version', 'Unknown')}")
        
        # Database health
        st.subheader("🗃️ Database Health")
        db_data = detailed_data.get("database", {})
        
        if db_data.get("status") == "connected":
            st.success(f"✅ Database connected ({db_data.get('response_time_ms', 0):.1f}ms)")
        else:
            st.error(f"❌ Database error: {db_data.get('message', 'Unknown')}")
        
        # System metrics
        st.subheader("💻 System Metrics")
        system_data = detailed_data.get("system", {})
        
        if "cpu_percent" in system_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CPU Usage", f"{system_data.get('cpu_percent', 0):.1f}%")
            
            with col2:
                st.metric("Memory Usage", f"{system_data.get('memory_percent', 0):.1f}%")
            
            with col3:
                st.metric("Available Memory", f"{system_data.get('available_memory_mb', 0):.0f} MB")
        
        # ML Model health
        st.subheader("🤖 ML Model Health")
        ml_data = detailed_data.get("ml_model", {})
        
        if ml_data.get("status") == "loaded":
            st.success("✅ ML model loaded successfully")
            
            if "test_prediction" in ml_data:
                test_data = ml_data["test_prediction"]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Test Sentiment:** {test_data.get('sentiment', 'N/A')}")
                
                with col2:
                    st.write(f"**Response Time:** {test_data.get('response_time_ms', 0):.1f}ms")
        else:
            st.error(f"❌ ML model error: {ml_data.get('message', 'Unknown')}")
    
    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Status"):
        st.experimental_rerun()

def main():
    """Main application function."""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        show_login_form()
    else:
        show_main_dashboard()

if __name__ == "__main__":
    main()