# CoSMIC Routing Benchmark — Complete Guide

This guide explains how to run the routing benchmark on top of an existing CoSMIC installation.
The benchmark measures how accurately CoSMIC's orchestrator selects the correct service
given a set of input prompts, **without calling any real service** (no chess engine, no vector DB, no code generator).

---

## Table of Contents

1. [What This Benchmark Does](#1-what-this-benchmark-does)
2. [File Overview](#2-file-overview)
3. [Prerequisites](#3-prerequisites)
4. [First-Time Setup](#4-first-time-setup)
5. [Preparing Your CSV File](#5-preparing-your-csv-file)
6. [Configuring Services](#6-configuring-services)
7. [Running the Benchmark](#7-running-the-benchmark)
8. [Reading the Output](#8-reading-the-output)
9. [Changing the Dataset](#9-changing-the-dataset)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What This Benchmark Does

CoSMIC normally works like this:

```
User types a question
        ↓
CoSMIC analyses the question and picks a service (chess / code / QA / ...)
        ↓
CoSMIC calls that service and returns the result
```

The benchmark replaces the first two steps only:

```
CSV file (column A = question)
        ↓
CoSMIC analyses the question and picks a service  ← same logic, unchanged
        ↓
Prints the selected service name  ← instead of calling the real service
```

The routing logic (how CoSMIC thinks) is **identical** to production.
Only the input source (CSV instead of a user) and the output (print instead of execute) are different.

---

## 2. File Overview

These are the only 3 files modified from the original CoSMIC codebase:

```
CoSMIC-production/
├── src/
│   ├── query_analyser/
│   │   ├── query_analyser.py      ← MODIFIED: benchmark runner added at the bottom
│   │   └── user_prompt.py         ← MODIFIED: SERVICES dict added, default value added
│   └── services/
│       └── llms/
│           └── prompts/
│               └── system_prompt.py  ← MODIFIED: ServiceSelectorPrompt class added
```

### What was changed and why

| File | What changed | Why |
|------|-------------|-----|
| `user_prompt.py` | Added `SERVICES` dict with dataset names and descriptions | To tell the LLM which services exist and what they cover |
| `user_prompt.py` | Added `= SERVICES` as default value in `__init__` | So the benchmark can call `QueryAnalyserService()` without passing services manually |
| `system_prompt.py` | Added `ServiceSelectorPrompt` class | To instruct the LLM to return only a service tag, nothing else |
| `query_analyser.py` | Added benchmark runner below the original `QueryAnalyser` class | To read from CSV, call the routing logic, and print the result |

**Everything else in these files is original CoSMIC code — untouched.**

---

## 3. Prerequisites

### Software required on your machine

- **Docker Desktop** — [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Git Bash** (Windows) — included with Git for Windows: [https://git-scm.com](https://git-scm.com)
- A working CoSMIC installation (see the [official CoSMIC README](https://github.com/TheOpenSI/CoSMIC))

### CoSMIC must already be running

Before running the benchmark, make sure CoSMIC is up:

```bash
# Run this in PowerShell or Git Bash
docker ps
```

You should see both `cosmic` and `ollama` in the list with status `Up`.

If not, start CoSMIC first:

```bash
bash start.sh
```

---

## 4. First-Time Setup

This section only needs to be done **once**.

### Step A — Download the 3 modified files from Claude

Download these 3 files:
- `user_prompt.py`
- `system_prompt.py`
- `query_analyser.py`

Place them in the correct folders on your Windows machine:

```
D:/CosMIC OpenSI Capstone/CoSMIC-production/
├── src/
│   ├── query_analyser/
│   │   ├── query_analyser.py      ← place here
│   │   └── user_prompt.py         ← place here
│   └── services/
│       └── llms/
│           └── prompts/
│               └── system_prompt.py  ← place here
```

---

### Step B — Copy all 3 files into the Docker container

Open **PowerShell** and run these 3 commands one by one:

```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/query_analyser/user_prompt.py" cosmic:/app/src/query_analyser/user_prompt.py
```

```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/services/llms/prompts/system_prompt.py" cosmic:/app/src/services/llms/prompts/system_prompt.py
```

```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/query_analyser/query_analyser.py" cosmic:/app/src/query_analyser/query_analyser.py
```

Each command should return:
```
Successfully copied XX.XkB to cosmic:/app/src/...
```

> If you see `Successfully copied 0B`, the file on Windows is empty. Re-download it from Claude and try again.

---

### Step C — Verify the setup

Open **Git Bash** and run:

```bash
docker exec -it cosmic bash
```

Then inside the container, run:

```bash
python -c "from src.query_analyser.user_prompt import SERVICES; print(list(SERVICES.keys()))"
```

You should see your list of services printed, for example:
```
['abstract_algebra', 'anatomy', 'astronomy', 'generic_qa']
```

If you see an error, go to the [Troubleshooting](#10-troubleshooting) section.

---

## 5. Preparing Your CSV File

The benchmark reads questions from a CSV file. The file must follow this format:

### Format rules

- **No header row** — the first row is already data
- **Column A (index 0)** — the question / prompt
- **Column B (index 1)** — the expected (correct) service name *(optional, only needed for accuracy calculation)*

### Example

```
Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the index of <p> in S_5.,abstract_algebra
A "dished face" profile is often associated with?,anatomy
Why is the sky blue?,astronomy
```

### Naming convention

Name your file descriptively so you can track which dataset was used:

```
Dataset3.csv       ← 3 services
Dataset5.csv       ← 5 services
Dataset10.csv      ← 10 services
merged_output_20260409.csv
```

---

## 6. Configuring Services

The list of services the LLM chooses from is defined in `user_prompt.py` inside the `SERVICES` dict.

### Current services

Open `src/query_analyser/user_prompt.py` and find:

```python
SERVICES = {
    "abstract_algebra":         "Contains theoretical mathematics problems...",
    "anatomy":                  "Includes questions about human body...",
    "astronomy":                "Covers celestial objects and space-related concepts...",
    "business_ethics":          "Consists of ethical decision-making scenarios...",
    "clinical_knowledge":       "Covers patient-centered medical scenarios...",
    "college_biology":          "Focuses on theoretical and conceptual biology...",
    "college_chemistry":        "Includes chemistry problems...",
    "college_computer_science": "Covers algorithms and data structures...",
    "mathematics":              "Contains general math problems...",
    "medicine":                 "Includes broad medical knowledge questions...",
    "generic_qa":               "Fallback service for ambiguous queries...",
}
```

### How to enable / disable a service

```python
SERVICES = {
    "abstract_algebra": "Contains theoretical mathematics...",   # ← enabled
    # "anatomy":        "Includes questions about body...",      # ← disabled (commented out)
    "generic_qa":       "Fallback service...",                   # ← enabled
}
```

- Add `# ` at the start of a line → service is **disabled**
- Remove `# ` from the start → service is **enabled**

### How to add a new service

Simply add a new line:

```python
SERVICES = {
    "abstract_algebra": "Contains theoretical mathematics...",
    "my_new_service":   "Description of what this service handles.",   # ← new
}
```

### After changing services — re-copy the file

```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/query_analyser/user_prompt.py" cosmic:/app/src/query_analyser/user_prompt.py
```

---

## 7. Running the Benchmark

### Step 1 — Copy your CSV file into Docker

Open **PowerShell**:

```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/YOUR_FILE.csv" cosmic:/app/YOUR_FILE.csv
```

Replace `YOUR_FILE.csv` with your actual filename.

---

### Step 2 — Enter the Docker container

Open **Git Bash**:

```bash
docker exec -it cosmic bash
```

You will see: `root@XXXXXXXXX:/app#`

---

### Step 3 — Run the benchmark

**Without accuracy** (just see what service was selected):

```bash
python src/query_analyser/query_analyser.py \
    --csv /app/YOUR_FILE.csv \
    --gt-col -1 \
    --use-example 2>&1
```

**With accuracy** (compare selected vs expected, CSV must have column B):

```bash
python src/query_analyser/query_analyser.py \
    --csv /app/YOUR_FILE.csv \
    --gt-col 1 \
    --use-example 2>&1
```

---

### All available arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv` | Path to the CSV file **inside the container** (always starts with `/app/`) | Required |
| `--gt-col` | Column index of the expected service. Use `1` if column B has the answer. Use `-1` to skip accuracy | `-1` |
| `--use-example` | Adds one example question/answer to the prompt — helps the LLM understand the format | Off |
| `--delay` | Seconds to wait between LLM calls — increase if Ollama is slow | `0.1` |

---

## 8. Reading the Output

### Without accuracy

```
[INFO] Queries  : 100
[INFO] Services : abstract_algebra, anatomy, astronomy, generic_qa
[INFO] Model    : mistral:latest

--------------------------------------------------------------------------------
#      SELECTED SERVICE                 PROMPT (first 60 chars)
--------------------------------------------------------------------------------
1      abstract_algebra                 Let p = (1, 2, 5, 4)(2, 3) in S_5...
2      abstract_algebra                 Find all zeros in the indicated fini...
3      anatomy                          A "dished face" profile is often ass...
4      generic_qa                       What would weigh most on the moon?...
--------------------------------------------------------------------------------
[DONE] 100 queries processed.
```

### With accuracy

```
--------------------------------------------------------------------------------
#      SELECTED                         EXPECTED                         MATCH
--------------------------------------------------------------------------------
1      abstract_algebra                 abstract_algebra                 ✓
2      abstract_algebra                 abstract_algebra                 ✓
3      astronomy                        anatomy                          ✗
4      generic_qa                       generic_qa                       ✓
--------------------------------------------------------------------------------
[DONE] 100 queries processed.
[DONE] Accuracy : 87/100 = 87.0%
```

- `✓` — CoSMIC selected the correct service
- `✗` — CoSMIC selected the wrong service

---

## 9. Changing the Dataset

Every time you want to run a different CSV file, you only need:

**PowerShell** — copy the new file:
```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/NEW_FILE.csv" cosmic:/app/NEW_FILE.csv
```

**Git Bash inside Docker** — run with the new file:
```bash
python src/query_analyser/query_analyser.py \
    --csv /app/NEW_FILE.csv \
    --gt-col -1 \
    --use-example 2>&1
```

If you also changed the services list, re-copy `user_prompt.py` as well (see [Step 2 in First-Time Setup](#step-b--copy-all-3-files-into-the-docker-container)).

---

## 10. Troubleshooting

### Container keeps restarting

**Symptom:** `docker exec -it cosmic bash` returns `Error response from daemon: Container is restarting`

**Cause:** `query_analyser.py` has a broken import that crashes CoSMIC on startup.

**Fix:**
```bash
# Restore the original file
docker exec cosmic bash -c "cd /app && git checkout src/query_analyser/query_analyser.py"

# Wait a few seconds for the container to recover, then re-copy the correct file
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/query_analyser/query_analyser.py" cosmic:/app/src/query_analyser/query_analyser.py
```

---

### ImportError: cannot import name 'SERVICES'

**Cause:** The old `user_prompt.py` is still in the container.

**Fix:**
```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/query_analyser/user_prompt.py" cosmic:/app/src/query_analyser/user_prompt.py
```

---

### ImportError: cannot import name 'ServiceSelectorPrompt'

**Cause:** The old `system_prompt.py` is still in the container.

**Fix:**
```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/services/llms/prompts/system_prompt.py" cosmic:/app/src/services/llms/prompts/system_prompt.py
```

---

### No such file or directory (CSV)

**Cause:** The CSV file was not copied into Docker yet.

**Fix:**
```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/YOUR_FILE.csv" cosmic:/app/YOUR_FILE.csv
```

---

### Successfully copied 0B

**Cause:** The file on Windows is empty — it was not saved correctly after downloading.

**Fix:** Re-download the file from Claude, make sure it has content, then run `docker cp` again.

---

### Script runs but prints nothing

**Cause:** Stale `__pycache__` files are loading old compiled versions of the code.

**Fix** (run inside Docker):
```bash
find /app/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "Cache cleared"
```

Then run the benchmark again.

---

### Read timed out

**Cause:** Ollama is still loading the model into memory.

**Fix:** Wait 30 seconds and try again. If it keeps happening, the model may be too large for your available RAM. Check Docker Desktop → Settings → Resources and increase the memory limit.

---

### unrecognized arguments: --output

**Cause:** An older version of `query_analyser.py` is still in the container (the version without `--output` support).

**Fix:**
```bash
docker cp "D:/CosMIC OpenSI Capstone/CoSMIC-production/src/query_analyser/query_analyser.py" cosmic:/app/src/query_analyser/query_analyser.py
```
