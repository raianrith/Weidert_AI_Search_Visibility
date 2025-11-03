#!/usr/bin/env python3
"""
Test script for Google Sheets integration
Run this to verify your setup before using the Streamlit app
"""

import sys
import os

def test_imports():
    """Test if required packages are installed"""
    print("=" * 50)
    print("Testing Package Installation")
    print("=" * 50)
    print()
    
    try:
        import gspread
        print("✅ gspread is installed")
    except ImportError:
        print("❌ gspread is NOT installed")
        print("   Run: pip install gspread")
        return False
    
    try:
        from google.oauth2.service_account import Credentials
        print("✅ google-auth is installed")
    except ImportError:
        print("❌ google-auth is NOT installed")
        print("   Run: pip install google-auth google-auth-oauthlib google-auth-httplib2")
        return False
    
    try:
        import streamlit
        print("✅ streamlit is installed")
    except ImportError:
        print("❌ streamlit is NOT installed")
        print("   Run: pip install streamlit")
        return False
    
    print()
    return True


def test_credentials():
    """Test if credentials file exists and is valid"""
    print("=" * 50)
    print("Testing Credentials")
    print("=" * 50)
    print()
    
    secrets_path = ".streamlit/secrets.toml"
    
    if not os.path.exists(secrets_path):
        print(f"❌ {secrets_path} does NOT exist")
        print(f"   Please create this file and add your credentials")
        return False
    
    print(f"✅ {secrets_path} exists")
    
    # Try to read and parse the file
    try:
        with open(secrets_path, 'r') as f:
            content = f.read()
            
        if 'gcp_service_account' not in content:
            print("❌ gcp_service_account section NOT found in secrets.toml")
            return False
        
        print("✅ gcp_service_account section found")
        
        # Check for required fields
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        missing_fields = []
        
        for field in required_fields:
            if field not in content:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required fields: {', '.join(missing_fields)}")
            return False
        
        print("✅ All required fields present")
        
    except Exception as e:
        print(f"❌ Error reading secrets.toml: {e}")
        return False
    
    print()
    return True


def test_connection():
    """Test actual connection to Google Sheets API"""
    print("=" * 50)
    print("Testing Google Sheets Connection")
    print("=" * 50)
    print()
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        import toml
        
        # Read credentials
        with open('.streamlit/secrets.toml', 'r') as f:
            secrets = toml.load(f)
        
        if 'gcp_service_account' not in secrets:
            print("❌ gcp_service_account not found in secrets")
            return False
        
        creds_dict = secrets['gcp_service_account']
        
        print(f"Service account email: {creds_dict.get('client_email', 'N/A')}")
        print(f"Project ID: {creds_dict.get('project_id', 'N/A')}")
        print()
        
        # Define scope
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Create credentials
        print("Creating credentials...")
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        print("✅ Credentials created successfully")
        
        # Authorize client
        print("Authorizing client...")
        client = gspread.authorize(creds)
        print("✅ Client authorized successfully")
        
        # Try to list spreadsheets (this will fail if APIs aren't enabled)
        print("Testing API access...")
        try:
            # This will trigger an error if APIs aren't enabled
            spreadsheets = client.openall()
            print(f"✅ API access successful! Found {len(spreadsheets)} accessible spreadsheets")
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                print("❌ API access forbidden - APIs might not be enabled")
                print()
                print("Please enable these APIs in Google Cloud Console:")
                print(f"  - Google Sheets API: https://console.cloud.google.com/apis/library/sheets.googleapis.com?project={creds_dict.get('project_id', 'YOUR_PROJECT')}")
                print(f"  - Google Drive API: https://console.cloud.google.com/apis/library/drive.googleapis.com?project={creds_dict.get('project_id', 'YOUR_PROJECT')}")
                return False
            else:
                raise
        
        print()
        print("🎉 All tests passed! Google Sheets integration is ready to use!")
        return True
        
    except FileNotFoundError:
        print("❌ Could not find .streamlit/secrets.toml")
        return False
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Run: pip install toml")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Common issues:")
        print("  1. Make sure Google Sheets API is enabled")
        print("  2. Make sure Google Drive API is enabled")
        print("  3. Check that your service account credentials are correct")
        return False


def main():
    """Run all tests"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Weidert LLM Tool - Google Sheets Connection Test".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    all_passed = True
    
    # Test 1: Package installation
    if not test_imports():
        all_passed = False
        print("⚠️  Please install missing packages before continuing")
        print()
    
    # Test 2: Credentials file
    if all_passed:
        if not test_credentials():
            all_passed = False
            print("⚠️  Please fix credentials configuration before continuing")
            print()
    
    # Test 3: Connection
    if all_passed:
        if not test_connection():
            all_passed = False
    
    # Summary
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print()
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("You're ready to use the Google Sheets upload feature!")
        print("Run your Streamlit app: streamlit run app.py")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Please fix the issues above and run this test again.")
        print()
        print("Need help? Check SETUP_GOOGLE_SHEETS.md for detailed instructions")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

