#!/usr/bin/env python3
"""
Test script for NVIDIA API integration
Run this to verify that NVIDIA models are properly configured
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from open_webui.routers.nvidia import NVIDIA_MODELS, get_model_info, get_nvidia_api_key

def test_environment_variables():
    """Test if environment variables are set"""
    print("🔍 Testing Environment Variables:")
    print("-" * 40)
    
    required_vars = [
        "NVIDIA_QWEN_API_KEY",
        "NVIDIA_KIMI_API_KEY", 
        "NVIDIA_DEEPSEEK_API_KEY"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value[:20]}...")
        else:
            print(f"❌ {var}: Not set")
    
    print()

def test_model_configuration():
    """Test model configuration"""
    print("🤖 Testing Model Configuration:")
    print("-" * 40)
    
    for model in NVIDIA_MODELS:
        print(f"Model ID: {model['id']}")
        print(f"  Name: {model['name']}")
        print(f"  API Model: {model['model_name']}")
        print(f"  API Key Env: {model['api_key_env']}")
        
        # Test if API key is available
        api_key = os.getenv(model['api_key_env'])
        if api_key:
            print(f"  ✅ API Key: Available ({api_key[:20]}...)")
        else:
            print(f"  ❌ API Key: Missing")
        print()

def test_model_matching():
    """Test model ID matching logic"""
    print("🔗 Testing Model Matching:")
    print("-" * 40)
    
    test_cases = [
        "nvidia/qwen3-coder-480b-a35b-instruct",
        "qwen3-coder-480b-a35b-instruct", 
        "qwen/qwen3-coder-480b-a35b-instruct",
        "nvidia/deepseek-r1-0528",
        "deepseek-ai/deepseek-r1-0528"
    ]
    
    for test_id in test_cases:
        model_info = get_model_info(test_id)
        if model_info:
            print(f"✅ {test_id} → {model_info['model_name']}")
        else:
            print(f"❌ {test_id} → Not found")
    print()

def test_api_key_retrieval():
    """Test API key retrieval for models"""
    print("🔑 Testing API Key Retrieval:")
    print("-" * 40)
    
    test_models = [
        "nvidia/qwen3-coder-480b-a35b-instruct",
        "nvidia/deepseek-r1-0528",
        "nvidia/moonshotai-kimi-k2-instruct-0905"
    ]
    
    for model_id in test_models:
        try:
            api_key = get_nvidia_api_key(model_id)
            print(f"✅ {model_id}: API key retrieved ({api_key[:20]}...)")
        except Exception as e:
            print(f"❌ {model_id}: {str(e)}")
    print()

def main():
    """Run all tests"""
    print("🚀 NVIDIA API Integration Test")
    print("=" * 50)
    print()
    
    test_environment_variables()
    test_model_configuration()
    test_model_matching()
    test_api_key_retrieval()
    
    print("✨ Test completed!")
    print()
    print("📝 Next steps:")
    print("1. Set missing environment variables in your .env file")
    print("2. Restart the backend server")
    print("3. Check that NVIDIA models appear in the model selector")

if __name__ == "__main__":
    main()
