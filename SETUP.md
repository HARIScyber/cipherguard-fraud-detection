# 🚀 CipherGuard Setup Guide

## Installation & Configuration

### 1️⃣ **Install CyborgDB SDK**

```bash
# Install CyborgDB packages
pip install cyborgdb
pip install cyborgdb-service
```

### 2️⃣ **Set Environment Variables**

**Windows (cmd):**
```cmd
set CYBORGDB_API_KEY=cyborg_e3652dfedfa64a2392d9a927211ffd77
set CYBORGDB_CONNECTION_STRING=
```

**Windows (PowerShell):**
```powershell
$env:CYBORGDB_API_KEY = "cyborg_e3652dfedfa64a2392d9a927211ffd77"
$env:CYBORGDB_CONNECTION_STRING = ""
```

**Linux/macOS (Bash):**
```bash
export CYBORGDB_API_KEY="cyborg_e3652dfedfa64a2392d9a927211ffd77"
export CYBORGDB_CONNECTION_STRING=""
```

Or create `.env` file:
```bash
cp .env.example .env
```

Edit `.env`:
```
CYBORGDB_API_KEY=cyborg_e3652dfedfa64a2392d9a927211ffd77
CYBORGDB_CONNECTION_STRING=
CYBORGDB_SERVICE_URL=http://localhost:8000
```

### 3️⃣ **Install Project Dependencies**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4️⃣ **Run the API Server**

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
=== CipherGuard Fraud Detection System Starting ===
✅ CyborgDB SDK initialized
Mode: CyborgDB SDK
API ready for requests
```

---

## 📊 API Endpoints

### Test Fraud Detection

```bash
# Basic transaction (normal)
curl -X POST http://localhost:8001/detect \
  -H "Content-Type: application/json" \
  -d "{\"amount\": 100, \"merchant\": \"Amazon\", \"device\": \"desktop\", \"country\": \"US\"}"

# High-risk transaction (suspicious)
curl -X POST http://localhost:8001/detect \
  -H "Content-Type: application/json" \
  -d "{\"amount\": 15000, \"merchant\": \"Unknown\", \"device\": \"mobile\", \"country\": \"CN\"}"

# Check system stats
curl http://localhost:8001/stats

# Health check
curl http://localhost:8001/health
```

---

## 🔧 Configuration Details

### CyborgDB API Key
**Your API Key:** `cyborg_e3652dfedfa64a2392d9a927211ffd77`

This key enables:
- ✅ Client-side encryption
- ✅ Encrypted kNN search
- ✅ Secure vector storage
- ✅ Privacy-preserving fraud detection

### Connection String
Leave `CYBORGDB_CONNECTION_STRING` empty for default PostgreSQL backend.

For custom setup:
```
postgresql://user:password@host:port/database
```

---

## ⚠️ Fallback Modes

If CyborgDB SDK is unavailable:
1. ✅ API still works with **local shim** (in-memory storage)
2. ✅ All endpoints functional
3. ⚠️ No encryption (development only)

Check server logs:
- `✅ CyborgDB SDK initialized` → Production mode
- `⚠️ Using local shim` → Development mode

---

## 📈 Next Steps

1. **Load sample transactions** → Train anomaly detection model
2. **Test fraud detection** → Verify API responses
3. **Submit feedback** → Retrain model for accuracy
4. **Deploy to production** → Use Docker/Kubernetes

---

## 🐛 Troubleshooting

### CyborgDB SDK not found
```bash
pip install --upgrade cyborgdb cyborgdb-service
```

### Connection refused
- Check if CyborgDB service is running
- Verify API key is correct
- Check firewall/network settings

### Vectors not storing
- Check `CYBORGDB_API_KEY` is set
- Verify PostgreSQL is running (if using custom connection)
- Check application logs for errors

---

## 📚 Documentation

- [CyborgDB Docs](https://cybergdb.io/docs)
- [FastAPI Docs](http://localhost:8001/docs)
- [API Reference](README_START.md)

---

**Ready to secure fraud detection!** 🛡️
