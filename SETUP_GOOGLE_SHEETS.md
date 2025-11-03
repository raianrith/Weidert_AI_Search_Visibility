# Google Sheets Setup Guide

## Quick Start

Follow these steps to get Google Sheets upload working:

### Step 1: Install Required Packages

Run this command in your terminal:

```bash
pip install gspread google-auth google-auth-oauthlib google-auth-httplib2
```

Or if you're using the full requirements:

```bash
pip install -r requirements.txt
```

### Step 2: Enable Google Cloud APIs

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project: **gen-lang-client-0389115937**
3. Enable these APIs:
   - **Google Sheets API**: https://console.cloud.google.com/apis/library/sheets.googleapis.com
   - **Google Drive API**: https://console.cloud.google.com/apis/library/drive.googleapis.com

### Step 3: Add Your API Keys

Edit `.streamlit/secrets.toml` and replace the placeholder API keys:

```toml
openai_api_key = "sk-..."           # Your actual OpenAI key
gemini_api_key = "AI..."            # Your actual Gemini key  
perplexity_api_key = "pplx-..."     # Your actual Perplexity key
```

### Step 4: Test the Connection

1. Run your Streamlit app: `streamlit run app.py`
2. Look at the sidebar on the left
3. Click the **"Test Google Sheets Connection"** button
4. You should see: ✅ Connection successful!

If you see an error, read it carefully - it will tell you exactly what's wrong.

### Step 5: Try Uploading

1. Go to Tab 1 (Multi-LLM Response Generator)
2. Run some queries
3. Click **"📊 Upload to Google Sheets"**
4. You should see a success message with a link to your spreadsheet

## Troubleshooting

### Error: "Missing packages"

**Solution:** Install the packages:
```bash
pip install gspread google-auth google-auth-oauthlib google-auth-httplib2
```

### Error: "API not enabled"

**Solution:** Go to Google Cloud Console and enable:
- Google Sheets API
- Google Drive API

Direct links:
- https://console.cloud.google.com/apis/library/sheets.googleapis.com?project=gen-lang-client-0389115937
- https://console.cloud.google.com/apis/library/drive.googleapis.com?project=gen-lang-client-0389115937

### Error: "Permission denied" or "Access forbidden"

**Solution:** Make sure both APIs are enabled (see above)

### The button does nothing / No error message appears

**Possible causes:**
1. **Packages not installed** - Install them using pip
2. **Streamlit cache issue** - Restart Streamlit (Ctrl+C and run again)
3. **Browser cache** - Refresh the page (Ctrl+R or Cmd+R)

### Spreadsheet created but I can't access it

The spreadsheet is created by the service account. To access it:

**Option 1:** The app tries to make it publicly viewable (anyone with link can view)

**Option 2:** Manually share it with yourself:
1. Open the link provided
2. Click "Request access" or
3. Use the service account email to share: `streamlit-sheets-sa@gen-lang-client-0389115937.iam.gserviceaccount.com`

## Your Service Account Details

- **Email:** `streamlit-sheets-sa@gen-lang-client-0389115937.iam.gserviceaccount.com`
- **Project:** `gen-lang-client-0389115937`

## Need Help?

1. **Check the error message** - The app now shows detailed error messages
2. **Test connection first** - Use the "Test Google Sheets Connection" button in the sidebar
3. **Verify APIs are enabled** - Both Sheets and Drive APIs must be enabled
4. **Check package installation** - Run `pip list | grep gspread`

## Success Indicators

When everything works, you'll see:
- ✅ Green checkmark in the test connection
- 🎈 Balloons animation when upload succeeds
- A clickable link to your new Google Sheet
- The sheet will contain all your query results with proper formatting

