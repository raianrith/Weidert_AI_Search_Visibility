#!/bin/bash

echo "================================================"
echo "Weidert LLM Tool - Package Installation"
echo "================================================"
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null
then
    echo "❌ Error: pip is not installed"
    echo "Please install Python and pip first"
    exit 1
fi

# Use pip3 if pip is not available
PIP_CMD="pip"
if ! command -v pip &> /dev/null; then
    PIP_CMD="pip3"
fi

echo "📦 Installing required packages..."
echo ""

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    $PIP_CMD install -r requirements.txt
else
    echo "❌ requirements.txt not found"
    echo "Installing packages manually..."
    
    # Install packages one by one
    $PIP_CMD install streamlit
    $PIP_CMD install openai
    $PIP_CMD install google-generativeai
    $PIP_CMD install pandas
    $PIP_CMD install matplotlib
    $PIP_CMD install nltk
    $PIP_CMD install seaborn
    $PIP_CMD install gspread
    $PIP_CMD install google-auth
    $PIP_CMD install google-auth-oauthlib
    $PIP_CMD install google-auth-httplib2
    $PIP_CMD install gspread-dataframe
    $PIP_CMD install python-dotenv
    $PIP_CMD install plotly
    $PIP_CMD install numpy
    $PIP_CMD install pyyaml
fi

echo ""
echo "================================================"
echo "✅ Installation complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Add your API keys to .streamlit/secrets.toml"
echo "2. Enable Google Sheets API and Google Drive API"
echo "   https://console.cloud.google.com/apis/library/sheets.googleapis.com?project=gen-lang-client-0389115937"
echo "   https://console.cloud.google.com/apis/library/drive.googleapis.com?project=gen-lang-client-0389115937"
echo "3. Run the app: streamlit run app.py"
echo "4. Test connection using the sidebar button"
echo ""
echo "For detailed setup instructions, see SETUP_GOOGLE_SHEETS.md"
echo "================================================"

