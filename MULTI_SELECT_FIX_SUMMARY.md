# Multi-Select Service Routing - Bug Fixes

## Summary of Issues Fixed

### **Problem 1: Capability Query Detection Logic**
**Location:** `query_analyser.py` lines 359-369

**Original Issue:**
```python
capability_keywords = ["what can you do", "what are your services", ...]
is_capability_query = any(keyword in query_lower for keyword in capability_keywords)
```
- Was using simple substring matching which could cause false positives
- Example: "How can you help me write code?" would incorrectly match "how can you help"
- Not leveraging the sophisticated regex-based detection already in `user_prompt.py`

**Fix:**
- Now uses the `user_prompter_system_info.__call__()` method which has:
  - Regex patterns for precise matching (e.g., `^how can you help(\s+me)?\??$`)
  - Standalone phrase detection for short queries
  - Word count limits to avoid false positives
  - Only triggers for genuine capability questions
  
```python
capability_check_result = self.user_prompter_system_info(query)
if capability_check_result == "YES":
    # Direct capability question detected via regex patterns
    return {"capability_query": "system_information"}, service_info_dict
```

**Benefits:**
- ✅ More accurate detection (no false positives)
- ✅ DRY principle - reusing existing sophisticated logic
- ✅ Consistent behavior between query_analyser and user_prompt
- ✅ Faster - no LLM call needed for common capability questions

---

### **Problem 2: Chess Service Routing Logic**
**Location:** `qa.py` line 103, `query_analyser.py` lines 406-415

**Original Issue:**
```python
if service_option.find("0.") > -1:
```
- This check failed when service_option was `"0"` (main chess service)
- `"0".find("0.")` returns -1, so chess queries were falling through to Service 3

**Fix in query_analyser.py:**
- Added logic to detect when service `"0"` is selected
- Automatically parse the query to determine if it's `"0.0"` (FEN) or `"0.1"` (moves)
- Update the selected_services dict with the correct sub-service

**Fix in qa.py:**
```python
if service_option.startswith("0.") or service_option == "0":
```
- Now catches both `"0"` and `"0.x"` formats
- Added error handling for missing FEN or moves data
- Added else clause for malformed chess queries

---

### **Problem 3: Service Type Consistency**
**Location:** `query_analyser.py` lines 400-420

**Original Issue:**
- System info relevance check was being applied to ALL services (including chess and vector DB)
- This caused unnecessary LLM calls for services that don't need it

**Fix:**
```python
elif option in ["2", "3"]:
    # Only check system info relevance for code gen and general Q&A
    self.llm.set_user_prompter(self.user_prompter_system_info)
    relevance_analysis = self.llm(query)[0]
    relevance = self.get_system_information_relevance(relevance_analysis)
```
- Now only Services 2 and 3 check for system information relevance
- Services 0 and 1 skip this check entirely

---

### **Problem 4: Capability Query Handling in QA**
**Location:** `qa.py` lines 107-112

**Added Feature:**
```python
if service_option == "capability_query":
    # Use LLM to explain the available services
    system_info = service_info_dict["system_information"]
    user_prompt = f"Based on this information about my capabilities: {system_info}, please provide a clear and helpful explanation of what services I can provide to the user."
    response, raw_response = self.llm(user_prompt, context="")
    return response, raw_response, retrieve_score
```
- Properly handles the capability query marker
- Uses LLM to generate natural explanations (not hardcoded)
- Returns early to avoid falling through to other services

---

## Test Cases to Verify

### ✅ Capability Queries (Should Work)
```python
"What can you do?"
"What are your services?"
"How can you help me?"
"What are your capabilities?"
```
**Expected:** LLM-generated explanation of available services

### ✅ Chess Queries (Should Work)
```python
"Predict next move: e4, e5, Nf3"  # Service 0.1
"Predict move: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Service 0.0
```
**Expected:** Stockfish predictions + LLM explanation

### ✅ Code Generation Queries (Should Work)
```python
"Write a Python function to sort a list"
"Generate code for a binary search"
```
**Expected:** PyCapsule (Service 2) response

### ✅ Vector DB Queries (Should Work)
```python
"Update database: Python is a programming language"
"Update database: data/document.pdf"
```
**Expected:** "Vector database updated."

### ✅ General Q&A (Should Work)
```python
"What is machine learning?"
"Explain quantum computing"
```
**Expected:** LLM response with optional RAG context

---

## Code Flow After Fix

```
User Query
    ↓
Is it a capability query? (keyword check)
    ↓ YES → Return capability_query marker → QA handles with LLM explanation
    ↓ NO
Check service_index config or LLM analysis
    ↓
Get selected_services dict
    ↓
For each service in selected_services:
    ├─ Service "0" → Parse to get "0.0" or "0.1" → Update dict
    ├─ Service "0.x" → Just parse for info extraction
    ├─ Service "1" → Parse for PDF/text info
    └─ Service "2" or "3" → Check system info relevance
    ↓
Return selected_services + service_info_dict
    ↓
QA.py routes based on first service key:
    ├─ "capability_query" → LLM explanation
    ├─ "0.0" or "0.1" → Chess (Stockfish)
    ├─ "1" → Vector DB update
    ├─ "2" → PyCapsule (code generation)
    └─ "3" or else → General Q&A (with optional RAG)
```

---

## What Was Preserved

✅ **Multi-service selection capability** - The infrastructure supports selecting multiple services
✅ **Capability explanation feature** - "What can you do?" queries work correctly
✅ **LLM-generated explanations** - Not hardcoded, uses LLM for natural responses
✅ **Config-driven service override** - `service_index` parameter still works
✅ **All existing service logic** - Chess, Vector DB, PyCapsule, General Q&A

---

## Potential Future Enhancements

1. **True multi-service execution:** Currently takes the first service only. Could iterate through all selected services.
2. **Service priority:** Define which service takes precedence when multiple are selected.
3. **Service combination:** Allow combining services (e.g., code generation + explanation).
4. **Better error messages:** More specific error messages for each failure mode.

---

## Testing Commands

Run the main script with test queries:
```bash
cd f:\cosmic-dev\CoSMIC
python main.py
```

Or test via API:
```bash
python api.py
# Then send requests to http://localhost:PORT
```

---

## Files Modified

1. **`src/query_analyser/query_analyser.py`**
   - Fixed capability query detection (lines 380-395)
   - Fixed chess service parsing logic (lines 406-421)
   - Fixed system info relevance checks (only for services 2 & 3)

2. **`src/services/qa.py`**
   - Added capability query handler (lines 107-112)
   - Fixed chess service detection (line 115)
   - Added error handling for missing chess data (lines 120, 132)
   - Added else clause for malformed chess queries (lines 144-146)

---

## Debugging Tips

If routing still doesn't work:

1. **Enable verbose mode:**
   ```python
   response, _, _ = opensi_cosmic(query, verbose=True)
   ```

2. **Check service selection:**
   ```python
   selected_services, info = query_analyser(query, verbose=True)
   print(f"Selected: {selected_services}")
   ```

3. **Add debug prints in qa.py:**
   ```python
   print(f"DEBUG: service_option = '{service_option}'")
   print(f"DEBUG: selected_services = {selected_services}")
   ```

4. **Check config.yaml:**
   ```yaml
   service: -1  # Should be -1 for auto-selection
   ```
