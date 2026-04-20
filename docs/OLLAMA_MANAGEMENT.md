# How to Manage Ollama (Teaching You to Fish 🎣)

## 🔍 Understanding Ollama

**What is it?**
- Ollama is a LOCAL server that runs AI models
- It runs as a background process on Windows
- Listens on port 11434 (http://localhost:11434)
- Loads models into RAM when needed

---

## ✅ Check if Ollama is Running

### Method 1: Check the Process
```powershell
# See if Ollama.exe is running
Get-Process -Name "ollama" -ErrorAction SilentlyContinue
```

**If running:** You'll see process details  
**If not running:** No output

---

### Method 2: Test the API
```powershell
# Try to connect to Ollama API
Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing
```

**If running:** Returns status 200  
**If not running:** Connection error

---

### Method 3: List Models (Best Way)
```powershell
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" list
```

**If running:** Shows your models (llama3.2:3b)  
**If not running:** Starts Ollama automatically and shows models

---

## 🚀 Start Ollama

### Option 1: Let it Auto-Start (Easiest)
Ollama automatically starts when you:
- Run `ollama list`
- Run `ollama run <model>`
- Make API requests

**Just use it - it starts itself!**

---

### Option 2: Manual Start
```powershell
# Start Ollama server in background
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" serve
```

**Note:** This keeps the terminal open. Press Ctrl+C to stop it later.

---

### Option 3: Windows Service (Advanced)
```powershell
# Check if Ollama service exists
Get-Service -Name "Ollama*"

# Start the service (if it exists)
Start-Service -Name "OllamaService"
```

**Note:** Ollama 0.14.2 might not install as a Windows Service by default.

---

## 🛑 Stop Ollama

### Method 1: Stop the Process (Recommended)
```powershell
# Find Ollama process
Get-Process -Name "ollama"

# Stop it
Stop-Process -Name "ollama" -Force
```

---

### Method 2: If Running in Terminal
If you started it with `ollama serve`:
- Press `Ctrl + C` in that terminal

---

### Method 3: Task Manager (GUI)
1. Open Task Manager (`Ctrl + Shift + Esc`)
2. Find "ollama.exe"
3. Right-click → End Task

---

## 🔄 Restart Ollama

### Quick Restart
```powershell
# Stop
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue

# Wait a moment
Start-Sleep -Seconds 2

# Start (will auto-start on next command)
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" list
```

---

## 📊 Check Ollama Status

### Simple Status Check Script
```powershell
# Save this as: check_ollama.ps1

Write-Host "Checking Ollama Status..." -ForegroundColor Cyan

# Check process
$process = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($process) {
    Write-Host "✅ Ollama is RUNNING" -ForegroundColor Green
    Write-Host "   PID: $($process.Id)"
    Write-Host "   Memory: $([math]::Round($process.WorkingSet64/1MB, 2)) MB"
} else {
    Write-Host "❌ Ollama is NOT running" -ForegroundColor Red
}

# Check models
Write-Host "`nModels:" -ForegroundColor Cyan
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" list
```

**Run it:**
```powershell
.\check_ollama.ps1
```

---

## 🎯 Common Scenarios

### Scenario 1: "Is Ollama running?"
```powershell
Get-Process -Name "ollama" -ErrorAction SilentlyContinue
```
- Output = Running
- No output = Not running

---

### Scenario 2: "Start Ollama for chatbot"
**You don't need to!** Just run your chatbot:
```bash
streamlit run chat_ui.py
```
Ollama auto-starts when the chatbot makes a request.

---

### Scenario 3: "Ollama is using too much RAM"
```powershell
# Stop Ollama to free RAM
Stop-Process -Name "ollama" -Force

# It will restart when you use it again
```

---

### Scenario 4: "Chatbot says 'Cannot connect to Ollama'"

**Fix:**
```powershell
# Quick restart
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" list
```

---

## 🔧 Advanced: Ollama Configuration

### Where is Ollama installed?
```
C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\
```

### Where are models stored?
```
C:\Users\AbulRahman Metwalley\.ollama\models\
```

### Check Ollama version:
```powershell
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" --version
```

---

## 📝 Quick Reference Commands

```powershell
# Status check
Get-Process -Name "ollama" -ErrorAction SilentlyContinue

# List models (auto-starts if needed)
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" list

# Stop Ollama
Stop-Process -Name "ollama" -Force

# Start Ollama (manual)
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" serve

# Test if API is responding
Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing
```

---

## 🎓 Key Takeaways

1. **Ollama auto-starts** - Usually you don't need to manually start it
2. **Check with `Get-Process`** - Simple way to see if running
3. **Stop with `Stop-Process`** - Force close when needed
4. **List models** - `ollama list` both checks status and shows models
5. **Don't worry too much** - It manages itself well!

---

## 🐛 Troubleshooting

**Problem:** "Ollama won't stop"
```powershell
# Force kill
Stop-Process -Name "ollama" -Force
```

**Problem:** "Ollama won't start"
```powershell
# Try running manually to see errors
& "C:\Users\AbulRahman Metwalley\AppData\Local\Programs\Ollama\ollama.exe" serve
```

**Problem:** "Port 11434 in use"
```powershell
# Find what's using the port
netstat -ano | findstr "11434"

# Kill that process (if it's Ollama)
Stop-Process -Id <PID> -Force
```

---

Now you know how to fish! 🎣

**Pro tip:** Usually you just start your chatbot and Ollama handles itself. Only manually manage it if:
- You need to free RAM
- Something went wrong
- You're debugging
