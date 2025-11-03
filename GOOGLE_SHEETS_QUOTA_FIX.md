# Google Sheets Storage Quota Fix

## The Problem

The service account has run out of storage space (15GB limit) because it was creating a new spreadsheet every time you uploaded results.

## The Solution (EASY - 2 minutes!)

Instead of the service account creating spreadsheets, **you'll create ONE spreadsheet in your own Google Drive** and the app will add new sheets to it.

---

## 📋 Step-by-Step Setup

### Step 1: Create a Google Sheet

1. Go to [Google Sheets](https://sheets.google.com)
2. Click **"+ Blank"** to create a new spreadsheet
3. Name it: **"Weidert LLM Results - Master"** (or any name you want)

### Step 2: Share it with the Service Account

1. In your new Google Sheet, click **"Share"** button (top right)
2. In the "Add people and groups" field, paste this email:
   ```
   streamlit-sheets-sa@gen-lang-client-0389115937.iam.gserviceaccount.com
   ```
3. Make sure the permission is set to **"Editor"**
4. Click **"Send"** (you can skip notification)

### Step 3: Copy the Google Sheet URL

1. Copy the URL from your browser address bar
   - It looks like: `https://docs.google.com/spreadsheets/d/1ABC...XYZ/edit`
2. Keep this URL handy!

### Step 4: Upload Results

When you upload results in the app, you have **two options**:

**OPTION A - Use Your Sheet (Recommended):**
1. Paste your Google Sheet URL in the text box
2. Click "📊 Upload to Google Sheets"
3. Each upload creates a new tab with a timestamp

**OPTION B - Use Master Sheet:**
1. Leave the URL box empty
2. The app will look for a sheet named "Weidert LLM Results - Master" shared with the service account
3. If found, it adds results as new tabs

---

## ✅ Benefits of This Approach

- ✅ Uses YOUR Google Drive storage (not service account's limited 15GB)
- ✅ All results organized in ONE spreadsheet with multiple tabs
- ✅ Easy to find and manage
- ✅ Each upload creates a timestamped tab (e.g., "2025-11-03_14-30-45")
- ✅ No more quota errors!

---

## 🎯 Quick Test

1. Create the sheet and share it with the service account (see steps above)
2. Run a query in the app
3. Paste your sheet URL in the text box
4. Click "📊 Upload to Google Sheets"
5. Open your Google Sheet - you'll see a new tab with your results! 🎉

---

## 🆘 Troubleshooting

### Error: "Could not open spreadsheet"
- **Solution**: Make sure you shared the sheet with the service account email
- **Check**: The email should have "Editor" permissions

### Error: "Master spreadsheet not found"
- **Solution**: Either paste your sheet URL in the text box, OR create a sheet named exactly "Weidert LLM Results - Master" and share it

### How do I organize old results?
- Each upload creates a new tab in your sheet
- You can rename tabs, delete old ones, or archive them as needed
- All tabs stay in one spreadsheet for easy access

---

## 📧 Service Account Email (for sharing)

```
streamlit-sheets-sa@gen-lang-client-0389115937.iam.gserviceaccount.com
```

Copy this email and paste it when sharing your Google Sheet!

