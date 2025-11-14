# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Prerequisites
1. 🔑 **Groq API Key** - Get from [Groq Console](https://console.groq.com/)
2. 📁 **GitHub Repository** - Upload your files to GitHub
3. 📧 **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io/)

### Steps to Deploy

#### 1. **Prepare Repository**
```bash
# Your repo should have these files:
app.py
backend.py
requirements.txt
descriptions.txt
.env.template  # Template for environment variables
README.md
```

**Important**: Don't commit `.env` file with real API keys to GitHub!

#### 2. **Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click "New app"
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Click "Deploy!"

#### 3. **Set Environment Variables**
In Streamlit Cloud settings (Advanced settings), add:
```
GROQ_API_KEY = your_actual_api_key_here
```

#### 4. **Alternative: Environment Variables in Secrets**
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_actual_api_key_here"
```

### Platform-Specific Instructions

#### 🟡 **Streamlit Cloud (Recommended)**
- ✅ Free tier available
- ✅ Auto-deployment from GitHub
- ✅ Built-in secrets management
- ⚠️ Limited to 1GB RAM

#### 🔵 **Heroku**
```bash
# Add Procfile:
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy:
heroku create your-app-name
git push heroku main
heroku config:set GROQ_API_KEY=your_key
```

#### 🟢 **Railway**
```bash
# Add railway.toml:
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
```

#### ⚫ **Render**
```bash
# Build Command: pip install -r requirements.txt
# Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### 🔧 **Configuration for Cloud**

The app is now configured for cloud deployment:
- ✅ Dynamic file paths (no hard-coded Windows paths)
- ✅ Environment variable support for API keys
- ✅ File upload fallback if data missing
- ✅ Error handling for missing dependencies

### 🧪 **Local Testing Before Deploy**
```bash
# Test locally first:
streamlit run app.py

# Check if it works without the original data path
# The app should either load descriptions.txt or show file uploader
```

### 📊 **Resource Requirements**
- **RAM**: ~500MB-1GB (depending on data size)
- **CPU**: Basic tier sufficient
- **Storage**: ~100MB for models + your data
- **Build time**: 3-5 minutes (first time)

### 🔒 **Security Notes**
1. ❌ **Never commit API keys to Git**
2. ✅ **Use environment variables**
3. ✅ **Use `.gitignore` for sensitive files**
4. ✅ **Consider rate limiting for public apps**

### 🐛 **Troubleshooting**

#### Common Issues:
1. **"No module named 'backend'"**
   - Ensure backend.py is in the same directory as app.py

2. **"Data file not found"**
   - Upload descriptions.txt using the file uploader
   - Or commit descriptions.txt to your repository

3. **"Memory limit exceeded"**
   - Reduce batch size in backend.py
   - Use smaller embedding model
   - Limit data size

4. **"API key invalid"**
   - Check environment variable setup
   - Verify API key is active

### ✅ **You're Ready to Deploy!**

Your FAISS-based app **CAN be deployed** because:
- ✅ Uses `faiss-cpu` (no GPU required)
- ✅ Local vector storage (no external vector DB)
- ✅ Streamlit-compatible architecture
- ✅ All dependencies are pip-installable

**Next Steps:**
1. Push your code to GitHub
2. Deploy on Streamlit Cloud
3. Set your Groq API key
4. Test your deployed app!

🎉 **Happy Deploying!**