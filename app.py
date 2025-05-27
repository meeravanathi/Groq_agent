import streamlit as st
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
import time
from agent import EcommerceAgent, customer_context_manager

# Load environment variables
load_dotenv()

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="E-commerce Customer Service Bot",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        flex-direction: row-reverse;
        color: black;
    }
    .chat-message.bot {
        background-color: #f5f5f5;
        color: black;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
    }
    .chat-message.user .avatar { background-color: #2196f3; }
    .chat-message.bot .avatar { background-color: #4caf50; }
    .chat-message .message { flex: 1; padding: 0 10px; }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: black;
    }
    .metric-card h3 {
        color: #4caf50;
        margin: 0;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        color: #666;
    }
    .groq-badge {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .model-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "agent" not in st.session_state:
        # Check for Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY in your .env file.")
            st.info("Get your free API key from: https://console.groq.com/keys")
            st.stop()
        
        # Initialize agent with Groq
        try:
            with st.spinner("Initializing Groq AI model... This should be quick!"):
                st.session_state.agent = EcommerceAgent(
                    groq_api_key=groq_api_key,
                )
        except Exception as e:
            st.error(f"❌ Failed to initialize AI agent: {str(e)}")
            st.info("Please check your Groq API key and internet connection.")
            st.stop()

    if "customer_authenticated" not in st.session_state:
        st.session_state.customer_authenticated = False

    if "current_customer" not in st.session_state:
        st.session_state.current_customer = None

    if "current_model" not in st.session_state:
        st.session_state.current_model = "llama3-70b-8192"

def display_message(role, content, timestamp=None):
    """Display a chat message with styling"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")

    avatar = "👤" if role == "user" else "🤖"
    css_class = "user" if role == "user" else "bot"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">
            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">
                {role.title()} • {timestamp}
            </div>
            <div>{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def process_user_message(prompt, context):
    """Process user message and get AI response"""
    try:
        with st.spinner("🤖 Groq AI is thinking..."):
            response = st.session_state.agent.process_message(prompt, context)
        return response
    except Exception as e:
        error_msg = f"⚠️ Sorry, I encountered an error: {str(e)}"
        if "rate limit" in str(e).lower():
            error_msg += "\n\n💡 You may have hit the API rate limit. Groq has generous free limits, please wait a moment and try again."
        elif "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_msg += "\n\n🔑 Please check your Groq API key in the .env file."
        elif "timeout" in str(e).lower():
            error_msg += "\n\n⏱️ Request timed out. Please try again."
        return error_msg

def main():
    """Main Streamlit app"""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🛍️ Customer Service")
      
        st.markdown("""
        <div class="groq-badge">
            ⚡ Powered by GROQ
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        # Model Selection
        st.subheader("🤖 AI Model Settings")
        available_models = st.session_state.agent.get_available_models()
        
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            index=available_models.index(st.session_state.current_model),
            help="Different models have varying capabilities and speeds"
        )
        
        if selected_model != st.session_state.current_model:
            if st.button("🔄 Switch Model", use_container_width=True):
                with st.spinner(f"Switching to {selected_model}..."):
                    if st.session_state.agent.switch_model(selected_model):
                        st.session_state.current_model = selected_model
                        st.success(f"✅ Switched to {selected_model}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to switch model")

        # Current model info
        model_info = {
            "llama3-70b-8192": "🦙 LLaMA 3 70B - Most capable",
            "llama3-8b-8192": "🦙 LLaMA 3 8B - Fast & efficient", 
            "mixtral-8x7b-32768": "🎯 Mixtral 8x7B - Large context",
            "gemma-7b-it": "💎 Gemma 7B - Instruction tuned"
        }
        
        st.info(f"**Current**: {model_info.get(st.session_state.current_model, 'Unknown')}")
        
        st.markdown("---")
        
        # Customer Login Section
        st.subheader("👤 Customer Login")

        if not st.session_state.customer_authenticated:
            login_method = st.radio("Login with:", ["Customer ID", "Email"])

            if login_method == "Customer ID":
                customer_id = st.text_input("Customer ID", placeholder="e.g., CUST001")
                if st.button("Login with ID", use_container_width=True):
                    if customer_id:
                        customer_context_manager.set_customer_id(st.session_state.session_id, customer_id)
                        st.session_state.current_customer = customer_id
                        st.session_state.customer_authenticated = True
                        st.rerun()
                    else:
                        st.warning("Please enter a Customer ID")
            else:
                email = st.text_input("Email", placeholder="your.email@example.com")
                if st.button("Login with Email", use_container_width=True):
                    if email and "@" in email:
                        customer_context_manager.set_customer_email(st.session_state.session_id, email)
                        st.session_state.current_customer = email
                        st.session_state.customer_authenticated = True
                        st.rerun()
                    else:
                        st.warning("Please enter a valid email address")
        else:
            st.success(f"✅ Logged in as:\n{st.session_state.current_customer}")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.customer_authenticated = False
                st.session_state.current_customer = None
                st.rerun()

        st.markdown("---")
        st.subheader("⚡ Quick Actions")

        quick_actions = {
            "📦 Check Order Status": "I'd like to check the status of my order",
            "🚚 Track Package": "Can you help me track my package?",
            "↩️ Return Item": "I need to return an item",
            "🔍 Product Search": "I'm looking for products",
            "💡 Get Recommendations": "Can you recommend some products?",
            "🌤️ Weather Update": "What's the weather like for shipping?"
        }

        for action, prompt in quick_actions.items():
            if st.button(action, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                context = customer_context_manager.get_context(st.session_state.session_id)
                response = process_user_message(prompt, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        st.markdown("---")
        st.subheader("📊 System Status")
        st.markdown("""
        <div class="status-card">
            <h4>🟢 All Systems Operational</h4>
            <p>✅ Order Management: Online<br>
            ✅ Product Database: Online<br>
            ✅ Weather Service: Online<br>
            ✅ Ollama API: Connected</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.reset_conversation()
            st.success("Chat cleared!")
            time.sleep(1)
            st.rerun()

    # Main content area
    st.title("🤖 E-commerce Customer Service Assistant")
    st.markdown("Welcome! I'm powered by **Groq's Lightning-Fast AI** ⚡. I'm here to help you with orders, products, returns, and more!")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><h3>⚡</h3><p>Lightning Fast</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><h3>9</h3><p>Tools Available</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><h3>~1s</h3><p>Avg Response</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><h3>24/7</h3><p>Available</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Chat interface
    chat_container = st.container()
    
    # Display existing messages
    with chat_container:
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="text-align: center;  background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="color: black">👋 Hello! How can I help you today?</h3>
                <p style="color: black">I'm running on <strong>{st.session_state.current_model}</strong> via Groq for ultra-fast responses!</p>
                <p style="color: black">Try asking about orders, products, returns, or use the quick actions in the sidebar!</p>
            </div>
            """, unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            display_message(message["role"], message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here... 💬"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with chat_container:
            display_message("user", prompt)

        # Get context and process message
        context = customer_context_manager.get_context(st.session_state.session_id)
        response = process_user_message(prompt, context)

        # Add and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with chat_container:
            display_message("assistant", response)

    # Help section
    with st.expander("💡 Sample Conversations & Tips"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🛒 Order Management:**
            - "What's the status of order ORD001?"
            - "I need to cancel order ORD002"
            - "I want to return the items from order ORD003"
            
            **🔍 Product Search:**
            - "Show me wireless headphones"
            - "I'm looking for electronics under $100"
            - "What are the details of product PROD001?"
            """)
        
        with col2:
            st.markdown("""
            **👤 Customer Service:**
            - "What are my loyalty points?"
            - "Update my preferred categories to include Books"
            - "Recommend products based on today's weather"
            
            **🔗 Complex Queries:**
            - "Check my order status and recommend similar products"
            - "What's the weather like and suggest appropriate clothing"
            """)

    # Groq-specific help
    with st.expander("⚡ About Groq Integration"):
        st.markdown("""
        **Why Groq?**
        - **Lightning Fast**: Responses in under 1 second
        - **Cost Effective**: Generous free tier
    
        **Available Models:**
        - **LLaMA 3 70B**: Most capable, best for complex queries
        - **LLaMA 3 8B**: Fast and efficient, great for quick responses  
        - **Gemma 7B**: Instruction-tuned for following directions
        
    
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em; padding: 1rem;">
        🤖 E-commerce Customer Service Bot • Powered by LangChain + Groq ⚡ • 
        Ultra-fast AI responses with enterprise reliability
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()