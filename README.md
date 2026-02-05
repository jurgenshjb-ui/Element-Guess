# 🧪 Element Guess

A Wordle-inspired guessing game based on the periodic table.

Play by clicking elements on the periodic table and using progressively revealed chemistry clues to narrow down the hidden element. Includes **Daily** and **Infinite** modes, plus **Easy / Normal / Hard** difficulty.

---

## ✨ Features

- **Clickable periodic table** (includes lanthanoids + actinoids strip)
- **Daily** challenge + **Infinite** mode
- **Clue system** (category, group family, electron block, period, atomic range, state, radioactivity, etc.)
- **Easy mode** highlights valid candidates
- **Normal mode** locks invalid candidates
- **Hard mode** rejects invalid guesses without explanation (no guess consumed)
- **Share results** (Wordle-style grid) + **Copy link** button
- Tooltips for elements + per-guess clue match feedback

> Note: Debug tools are available in development but hidden in production deployments.

---

## 🚀 Run locally (Windows)

### 1) Install Python packages
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
