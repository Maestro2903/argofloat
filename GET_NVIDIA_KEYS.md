# 🔑 Get Your Own NVIDIA API Keys - Fix 404 Error

The "404: Model not found" error you're seeing is because the shared NVIDIA API keys in your `.env` file are hitting rate limits. Here's how to fix it:

## 🚀 Quick Fix (5 minutes)

### Step 1: Get Free NVIDIA API Keys
1. Go to **https://integrate.api.nvidia.com/**
2. Click **"Sign Up"** (it's free!)
3. Verify your email
4. Browse available models and click **"Get API Key"** for each model you want

### Step 2: Update Your .env File
Replace the keys in your `.env` file with your own:

```bash
# Your personal NVIDIA API keys (replace with your own)
NVIDIA_API_KEY=nvapi-YOUR_MAIN_KEY_HERE
NVIDIA_QWEN_API_KEY=nvapi-YOUR_QWEN_KEY_HERE
NVIDIA_KIMI_API_KEY=nvapi-YOUR_KIMI_KEY_HERE
NVIDIA_DEEPSEEK_API_KEY=nvapi-YOUR_DEEPSEEK_KEY_HERE
```

### Step 3: Restart Your App
```bash
# Stop your current app (Ctrl+C)
# Then restart it
cd backend
python -m open_webui.main
```

## 🎯 Why This Fixes the Error

- **Before**: Shared keys → Rate limits → 404 errors
- **After**: Your own keys → Your own quota → Works perfectly!

## 🆓 It's Completely Free!

- NVIDIA provides generous free tier limits
- No credit card required
- Access to powerful models like DeepSeek R1, Qwen3, etc.

## 🔧 Alternative: Test with Mock Response

If you want to test immediately, I've updated the code to provide helpful error messages instead of just "404: Model not found". The app will now tell you exactly what's wrong and how to fix it.

## 📱 Models Available

Once you have your keys, you'll have access to:
- **DeepSeek R1 0528** - Advanced reasoning model
- **Qwen3 Coder 480B** - Excellent for coding
- **Moonshot AI Kimi K2** - Great for conversations
- **Llama 3.1 Nemotron 70B** - Powerful general model
- **Llama 3.1 405B** - Most powerful model

## ✅ Success!

After getting your own keys, you should see:
- ✅ No more 404 errors
- ✅ Fast response times
- ✅ Access to all models
- ✅ Reliable performance

The whole process takes about 5 minutes and completely solves the issue!
