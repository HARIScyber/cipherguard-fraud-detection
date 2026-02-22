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

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

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
        <h1>🔐 Sentiment Analysis Dashboard</h1>
        <h3>Please login to continue</h3>
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
        <h1>📊 Sentiment Analysis Dashboard</h1>
        <p>Welcome back, {st.session_state.user_info.get('full_name', 'User')}!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        selected_tab = st.selectbox(
            "Choose a section:",
            ["🏠 Overview", "🔍 Analyze Comments", "📈 Analytics", "📋 Comment History", "⚙️ System Status"]
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
    elif selected_tab == "📈 Analytics":
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