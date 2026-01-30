# 🚀 RevoScope - Quick Start Guide

Follow these steps to run the complete system with the newly trained **AST AI Model**.

### 🛠️ Prerequisites
- **Python 3.10+** (in `backend/` folder)
- **Node.js 18+** (in root folder)

---

### Step 1: Start the AI Backend (Terminal 1)
Open a terminal in the `backend/` directory:
```powershell
# 1. Activate the environment (from backend folder)
..\.venv\Scripts\activate

# Or if you are in the root folder:
# .\.venv\Scripts\activate

# 2. Run the AST-optimized server
python api_server_ast.py
```
*Wait until you see: **"✓ AST model loaded (77.0% accuracy)"***

### Step 2: Start the Frontend (Terminal 2)
Open a **new** terminal in the root project folder:
```bash
# 1. Install dependencies (only required once)
npm install

# 2. Start the Expo app
npm start
```

---

### ✅ How to Verify
1. **Model Check:** Open [http://localhost:8000/health](http://localhost:8000/health) - it should show `"model_type": "ast"`.
2. **App Usage:** Open the app on your phone (Expo Go) or press **'w'** in the terminal for the web version.
3. **Analyze:** Upload an audio file. The app will send it to the AST server for high-accuracy lung analysis.

### 💻 Setting up on a New Computer
If you clone this project onto a different machine, run these commands to get the 300MB+ AI model:

```powershell
# 1. Install & Initialize Git LFS
git lfs install

# 2. Download the actual AI model weights (if they are just pointers)
git lfs pull
```

---
### 📁 Important Files
- **Results:** [walkthrough.md](file:///C:/Users/User/.gemini/antigravity/brain/ca2df098-f262-4d47-899c-863aa79cc610/walkthrough.md)
- **Weights:** `backend/weights/respiratory_ast_best.pth` (Stored via LFS)
