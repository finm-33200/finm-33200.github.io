# Homework 0

**Setting up your computing environment**

This homework is not graded, but please complete it before the first class. You will need these tools installed and accounts created to complete the coursework.

---

## Software to Install

Install the following software on your laptop (all free):

### Core Development Tools

- **[Anaconda](https://www.anaconda.com/download)** - Python 3.10+ distribution with data science packages
- **[Visual Studio Code](https://code.visualstudio.com/)** - Code editor (NOT Visual Studio, which is different)
- **[Git](https://git-scm.com/)** - Version control
- **[GitKraken](https://www.gitkraken.com/)** - Git GUI (free for students via [GitHub Student Pack](https://www.gitkraken.com/github-student-developer-pack))

### AI Development Tools

- **[Claude Code](https://github.com/anthropics/claude-code)** - Anthropic's terminal AI agent
  ```bash
  # Requires Node.js 18+
  npm install -g @anthropic-ai/claude-code

  # Verify installation
  claude --version
  ```

- **[Cursor](https://cursor.sh/)** - AI-native code editor
  - Download from cursor.sh
  - Imports VS Code settings automatically

### Optional

- **[TeX Live](https://tug.org/texlive/)** - Only needed if you want to compile LaTeX documents

---

## Accounts to Create

Create accounts with the following services (free tiers available):

### Required

| Service | URL | Notes |
|---------|-----|-------|
| **GitHub** | [github.com](https://github.com/) | Code hosting, assignment submission |
| **OpenAI Platform** | [platform.openai.com](https://platform.openai.com/) | Add $5-10 API credits |
| **Anthropic Console** | [console.anthropic.com](https://console.anthropic.com/) | For Claude Code |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai/) | Multi-model API access (free tier available) |

### For Financial Data (Later in Course)

| Service | URL | Notes |
|---------|-----|-------|
| **WRDS** | [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu/) | Apply through UChicago ([registration](https://wrds-www.wharton.upenn.edu/register/)) |
| **IPUMS CPS** | [cps.ipums.org](https://cps.ipums.org/cps/) | Economic data |

For WRDS access issues, contact the UChicago Mathematics department representative: John Zekos (zekos@math.uchicago.edu).

---

## Environment Setup

### 1. Create .env File

API keys should never be committed to git. Create a `.env` file for local configuration:

```bash
# Create .env file in your project directory
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Clone Course Repository

```bash
git clone https://github.com/finm-33200/ai_inclass_examples.git
cd ai_inclass_examples
```

### 3. Install Python Packages

```bash
# Create conda environment
conda create -n finm33200 python=3.11
conda activate finm33200

# Install core packages
pip install openai python-decouple pydantic
```

### 4. Verify Setup

Test that your environment works:

```bash
cd ai_inclass_examples/basic_llm_api/02_openai_hello
python hello.py
```

You should see a response from GPT-4o-mini and token usage statistics.

---

## Verification Checklist

Before the first class, confirm:

- [ ] Python 3.10+ installed and working
- [ ] VS Code or Cursor installed
- [ ] Git installed (`git --version` works)
- [ ] Claude Code installed (`claude --version` works)
- [ ] GitHub account created
- [ ] OpenAI account with API credits
- [ ] Anthropic Console account
- [ ] OpenRouter account
- [ ] `.env` file created with API keys
- [ ] `ai_inclass_examples` repo cloned
- [ ] `hello.py` runs successfully

---

## Troubleshooting

### Claude Code Installation Issues

If `npm install -g` fails:

```bash
# On macOS, you may need to fix permissions
sudo chown -R $(whoami) ~/.npm

# Or use npx instead
npx @anthropic-ai/claude-code
```

### API Key Not Working

1. Verify the key is correct (no extra spaces)
2. Check that you have credits on the account
3. Ensure the `.env` file is in the correct directory

### Python Package Issues

```bash
# If packages fail to install, try upgrading pip first
pip install --upgrade pip

# Then retry installation
pip install openai python-decouple pydantic
```

---

## Additional Resources

See the [Appendix](appendix.md) for video tutorials on setting up Python and Visual Studio Code.
