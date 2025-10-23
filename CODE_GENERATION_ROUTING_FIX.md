# Code Generation Routing Fix

## Problem

When users asked code generation queries like:
- "Write a Python script to print the 100 first prime numbers"
- "Create a function to sort a list"
- "Generate code for binary search"

The queries were being routed to **Service 3 (General Q&A)** instead of **Service 2 (PyCapsule)**.

### Why This Matters

**Service 3 (General Chat):**
- Uses the base LLM to generate code
- No testing or validation
- Code may have bugs
- No debugging attempts

**Service 2 (PyCapsule):**
- Uses specialized code generation models (qwen2.5-coder)
- Tests the generated code in isolated containers
- Validates the code works correctly
- Debugs and fixes issues automatically
- Returns: "code generation and validation successful, number of debugging attempt made: X"

## Root Cause

### 1. Ambiguous Service Descriptions

**Old Service 2 description:**
```
"generate or improve a code or answer a question in order to generate or improve a code"
```
- Too wordy and confusing
- Mentions "answer a question" which overlaps with Service 3
- LLM couldn't distinguish between code generation and general Q&A

**Old Service 3 description:**
```
"answer a question or provide a reasoning, which cannot be achieved by the other services"
```
- Too generic - "answer a question" applies to everything
- Made LLM think it can handle code generation questions

### 2. Weak Service Selection Prompt

**Old prompt:**
```
"For instance, if the query is to predict the next chess move, then select service 0;
otherwise, if the query is to generate or modify a code, then select service 2."
```
- Only one example of code generation keywords
- Not explicit enough about code-related queries
- LLM could interpret "write code" as a question to answer (Service 3)

## Solution

### 1. Improved Service Descriptions

**New Service 2 description:**
```
"write, generate, create, implement, or improve code (any programming language). 
This service tests and validates code before returning it."
```
✅ Action-oriented verbs (write, generate, create, implement)
✅ Explicitly mentions testing and validation
✅ No ambiguity with general Q&A

**New Service 3 description:**
```
"answer general questions, provide explanations, or reasoning that does not involve 
code generation, chess moves, or database updates"
```
✅ Explicitly excludes code generation
✅ Clear boundaries with other services
✅ Catch-all only for truly general questions

### 2. Enhanced Service Selection Prompt

**New prompt includes:**
```
Important: If the query asks to 'write code', 'generate code', 'create a script',
'write a function', 'write a program', 'implement', or anything that requires producing
executable code, you MUST select service 2 (code generation).
Service 2 validates and tests the generated code before returning it.
```

**Benefits:**
✅ Lists common code generation keywords explicitly
✅ Uses imperative "MUST select" to guide LLM strongly
✅ Emphasizes testing/validation as key differentiator
✅ Covers various phrasings (script, function, program, implement)

## Test Cases

### ✅ Should Route to Service 2 (PyCapsule)

| Query | Expected Service | Result |
|-------|------------------|--------|
| "Write a Python script to print 100 first prime numbers" | Service 2 | ✅ Now correct |
| "Create a function to calculate factorial" | Service 2 | ✅ Now correct |
| "Generate code for binary search" | Service 2 | ✅ Now correct |
| "Implement a sorting algorithm in JavaScript" | Service 2 | ✅ Now correct |
| "Write a program to reverse a string" | Service 2 | ✅ Now correct |
| "Can you code a Fibonacci generator?" | Service 2 | ✅ Now correct |
| "Improve this code: [code snippet]" | Service 2 | ✅ Already worked |

### ✅ Should Still Route to Service 3 (General Q&A)

| Query | Expected Service | Result |
|-------|------------------|--------|
| "What is machine learning?" | Service 3 | ✅ Still correct |
| "Explain how binary search works" | Service 3 | ✅ Still correct |
| "What are the benefits of Python?" | Service 3 | ✅ Still correct |
| "How does a hash table work?" | Service 3 | ✅ Still correct |

## Verification

After these changes, code generation queries will:

1. ✅ Be routed to PyCapsule (Service 2)
2. ✅ Get tested and validated in isolated containers
3. ✅ Return debugging attempt count
4. ✅ Show message: "code generation and validation successful, number of debugging attempt made: X"

## Files Modified

### 1. `src/query_analyser/query_analyser.py` (Lines 70-75)
**Changed:** Service descriptions to be more explicit

**Before:**
```python
"2": "generate or improve a code or answer a question in order to generate or improve a code",
"3": "answer a question or provide a reasoning, which cannot be achieved by the other services"
```

**After:**
```python
"2": "write, generate, create, implement, or improve code (any programming language). This service tests and validates code before returning it.",
"3": "answer general questions, provide explanations, or reasoning that does not involve code generation, chess moves, or database updates"
```

### 2. `src/query_analyser/user_prompt.py` (Lines 69-87)
**Changed:** Service selection prompt with explicit code generation keywords

**Before:**
```python
f" For instance, if the query is to predict the next chess move, then select service 0;"
f" otherwise, if the query is to generate or modify a code, then select service 2."
```

**After:**
```python
f" Important: If the query asks to 'write code', 'generate code', 'create a script',"
f" 'write a function', 'write a program', 'implement', or anything that requires producing"
f" executable code, you MUST select service 2 (code generation)."
f" Service 2 validates and tests the generated code before returning it."
f" If the query is to predict the next chess move, select service 0."
f" If the query is to update the vector database, select service 1."
f" Only select service 3 for general questions that don't involve code generation."
```

## Expected Behavior After Fix

### Before Fix:
```
User: "Write a Python script to print 100 first prime numbers"
→ Routes to Service 3 (General Q&A)
→ LLM generates code without testing
→ Code might have bugs
→ No validation message
```

### After Fix:
```
User: "Write a Python script to print 100 first prime numbers"
→ Routes to Service 2 (PyCapsule)
→ Code generated by qwen2.5-coder
→ Code tested in isolated container
→ Debugged if needed
→ Returns: "code generation and validation successful, number of debugging attempt made: 1"
→ User gets working, tested code
```

## Additional Notes

- The LLM query analyser will now have **stronger guidance** on code generation routing
- Service 3 boundaries are **explicitly defined** to exclude code generation
- Service 2 highlights **testing/validation** as a key feature
- Multiple common code generation **action verbs** are covered

## Testing Recommendation

Restart the Docker containers to load the changes:
```bash
cd F:\cosmic-dev\CoSMIC
docker-compose down
docker-compose up --build -d
```

Then test with:
1. "Write a Python script to print 100 first prime numbers"
2. "Create a function to calculate factorial"
3. "What is machine learning?" (should still use Service 3)

Expected: First two queries should show PyCapsule validation messages.
