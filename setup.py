"""
Setup script for the E-commerce Customer Service Chatbot
"""
import os
import subprocess
import sys
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("""# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Weather API Configuration (Optional - OpenWeatherMap)
WEATHER_API_KEY=your_weather_api_key_here

# Application Configuration
APP_NAME=E-Commerce Customer Service Bot
DEBUG=True
""")
        print("✅ .env file created. Please add your API keys.")
    else:
        print("✅ .env file already exists.")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        "logs",
        "data",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_api_keys():
    """Check if API keys are configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    Tgt_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    weather_key = os.getenv("WEATHER_API_KEY")
    
    if not Tgt_key or Tgt_key == "your_groq_api_key_here":
        print("⚠️  Groq API key not configured. Please update your .env file.")
        return False
    
    if not weather_key or weather_key == "your_weather_api_key_here":
        print("⚠️  Weather API key not configured (optional). Weather features will use mock data.")
    
    print("✅ API keys are configured.")
    return True

def run_tests():
    """Run basic tests to ensure everything is working"""
    print("Running basic tests...")
    
    try:
        # Test imports
        from mock_databases import MockOrderDatabase, MockProductDatabase, MockCustomerDatabase
        from tools import get_tools
        from agent import EcommerceAgent
        
        print("✅ All imports successful.")
        
        # Test database initialization
        order_db = MockOrderDatabase()
        product_db = MockProductDatabase()
        customer_db = MockCustomerDatabase()
        
        print("✅ Mock databases initialized.")
        
        # Test tools
        tools = get_tools()
        print(f"✅ {len(tools)} tools loaded successfully.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up E-commerce Customer Service Chatbot...")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation.")
        return
    
    # Check API keys
    if not check_api_keys():
        print("⚠️  Please configure your API keys in the .env file before running the application.")
    
    # Run tests
    if run_tests():
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Update your .env file with valid API keys")
    print("2. Run the application: streamlit run app.py")
    print("3. Open your browser to the provided URL")
    print("\nFor help, check the README.md file.")

if __name__ == "__main__":
    main()
