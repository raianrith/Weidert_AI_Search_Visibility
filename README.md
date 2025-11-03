# Weidert Group LLM Search Visibility Tool

AI-Powered Competitive Intelligence for B2B Industrial Marketing. Analyze brand visibility across ChatGPT, Gemini & Perplexity.

## Features

- **Multi-LLM Response Generator**: Run queries across OpenAI, Gemini, and Perplexity simultaneously
- **Search Visibility Analysis**: Track mention rates, positions, context, and source citations
- **Competitor Comparison**: Head-to-head analysis with key competitors
- **Gap Analysis & Opportunities**: Identify visibility gaps and strategic content opportunities
- **Google Sheets Integration**: Export results directly to Google Sheets

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### API Keys

You need to configure API keys for the LLM services. Add them to Streamlit secrets or environment variables:

**Option 1: Streamlit Secrets** (`.streamlit/secrets.toml`):
```toml
openai_api_key = "your-openai-api-key"
gemini_api_key = "your-gemini-api-key"
perplexity_api_key = "your-perplexity-api-key"
```

**Option 2: Environment Variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export PERPLEXITY_API_KEY="your-perplexity-api-key"
```

### Google Sheets Integration Setup

To enable the "Upload to Google Sheets" feature, you need to set up Google Cloud credentials:

#### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Sheets API
   - Google Drive API

#### Step 2: Create Service Account Credentials

1. In Google Cloud Console, go to **IAM & Admin** > **Service Accounts**
2. Click **Create Service Account**
3. Give it a name (e.g., "Weidert LLM Tool")
4. Click **Create and Continue**
5. Skip the optional steps and click **Done**
6. Click on the service account you just created
7. Go to the **Keys** tab
8. Click **Add Key** > **Create New Key**
9. Choose **JSON** format
10. Download the JSON file

#### Step 3: Configure Streamlit Secrets

Add the Google Cloud credentials to your `.streamlit/secrets.toml` file:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account-email@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

**Note**: You can copy these values directly from the downloaded JSON file.

#### Step 4: Share Google Sheets Access

The service account needs access to create/edit Google Sheets. You have two options:

**Option A**: The tool will automatically create new spreadsheets, and they'll be owned by the service account. To access them:
- Share the spreadsheet with your Google account email after it's created
- Or, add your email to the service account's permissions

**Option B**: Pre-create a Google Sheet and share it with the service account email:
1. Create a new Google Sheet
2. Click "Share"
3. Add the service account email (found in the JSON file as `client_email`)
4. Give it "Editor" permissions

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

### Running Queries

1. **Multi-LLM Response Generator** tab:
   - Use the predefined 20 queries, or
   - Enter your own custom queries (one per line)
   - Click "Run" to generate responses from all three LLMs
   - Download results as CSV or upload to Google Sheets

2. **Search Visibility Analysis** tab:
   - Upload a CSV or use results from Tab 1
   - View mention rates, position analysis, context sentiment
   - Analyze source citations and domain rankings

3. **Competitor Comparison** tab:
   - Select competitors to analyze
   - View head-to-head performance metrics
   - See win/loss analysis and co-mention patterns

4. **Gap Analysis & Opportunities** tab:
   - Identify queries where Weidert is missing
   - See which competitors are winning in your gaps
   - Get prioritized action items

## Troubleshooting

### Google Sheets Upload Fails

- **"Missing credentials" error**: Ensure `gcp_service_account` is properly configured in secrets.toml
- **"Permission denied" error**: Share the Google Sheet with the service account email
- **"API not enabled" error**: Enable Google Sheets API and Google Drive API in Google Cloud Console

### API Rate Limits

If you encounter rate limit errors:
- Increase the "Delay Between Requests" slider in the sidebar
- Reduce the "Parallel Processing Workers" count
- Run fewer queries at once

## Support

For issues or questions, contact the Weidert Group development team.

## License

Proprietary - Weidert Group
