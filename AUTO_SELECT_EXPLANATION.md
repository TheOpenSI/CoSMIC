# Auto-Select Mode: How It Works

## Overview

When the user selects "Auto" (service_index=-1), the system uses an LLM to analyze the query and intelligently choose the most appropriate service.

## The Process

### Step 1: Check if Auto-Select Mode
```python
# In query_analyser.py, line 407
if self.service_index != ["-1"]:
    # Pre-configured service - use it directly
    ...
else:
    # Auto-select mode - continues below
```

When `service_index = -1` (Auto), the condition is FALSE, so we skip the pre-configured logic and continue to auto-select.

### Step 2: Build the LLM Prompt

The system uses `QueryAnalyserService` prompter to build a prompt for the LLM:

```python
# In query_analyser.py, lines 410-411
self.llm.set_user_prompter(self.user_prompter_service)
service_analysis = self.llm(query)[0]
```

**What the prompt looks like** (from `user_prompt.py`, lines 83-87):

```
Given 4 services: 'service 0: if it is a chess game, predict the next chess move by providing a sequence of moves or a FEN, service 1: update the vector database with a declarative sentence (not a question) or a pdf document, service 2: generate or improve a code or answer a question in order to generate or improve a code, service 3: answer a question or provide a reasoning, which cannot be achieved by the other services', which service can answer the following query? The query is '[USER_QUERY]'. For instance, if the query is to predict the next chess move, then select service 0; otherwise, if the query is to generate or modify a code, then select service 2. Just return which service without any explanations.
```

### Step 3: LLM Analyzes and Selects Service

The LLM (llama3.1:8b for query analysis) reads the prompt and returns something like:

**Example 1: Chess Query**
```
User Query: "Predict next move: e4, e5, Nf3"
LLM Response: "service 0"
```

**Example 2: Code Generation Query**
```
User Query: "Write a Python function to add two numbers"
LLM Response: "service 2"
```

**Example 3: General Question**
```
User Query: "What is the capital of France?"
LLM Response: "service 3"
```

### Step 4: Parse LLM Response

```python
# In query_analyser.py, lines 414-418
service_option = self.mapping(service_analysis)

# Always normalize to a list
if isinstance(service_option, str):
    service_option = [service_option]
```

The `mapping()` function (lines 145-172) uses regex to extract service numbers:
```python
# Find all mentions like "service 0", "service 1.0", etc.
options = re.findall(r'service (\d{1,3}(?:\.\d{1,3})?)', response)
# Returns: ["0"] or ["2"] or ["3"], etc.
```

### Step 5: Filter Valid Services

```python
# In query_analyser.py, lines 420-424
selected_services = {
    opt: self.full_services[opt]
    for opt in service_option
    if opt in self.full_services
}
```

This creates a dictionary like:
```python
# For chess: {"0": "if it is a chess game, predict the next chess move..."}
# For code gen: {"2": "generate or improve a code..."}
```

### Step 6: Fallback if No Valid Service

```python
# In query_analyser.py, lines 426-428
if not selected_services:
    selected_services = {"3": self.full_services["3"]}
```

If the LLM returns something invalid or no service is found, default to Service 3 (General Q&A).

### Step 7: Parse for Service-Specific Details

```python
# In query_analyser.py, lines 430-457
for option in list(selected_services.keys()):
    if option == "0":
        # Chess - parse to get FEN or moves
        parsed_option, service_info_dict = self.chess_parse(query, service_info_dict)
        # Update to specific sub-service: "0.0" or "0.1"
        
    elif option == "1":
        # Vector DB - parse to get text or document
        _, service_info_dict = self.update_vector_database_parse(query, service_info_dict)
        
    elif option in ["2", "3"]:
        # Check if query needs system information
        self.llm.set_user_prompter(self.user_prompter_system_info)
        relevance_analysis = self.llm(query)[0]
        # ... check relevance ...
```

## Complete Example Flow

### Example: Chess Query in Auto-Select Mode

**User Input:**
- Service Selection: Auto
- Query: "Predict next move: e4, e5, Nf3"

**Flow:**

1. **Capability Check** (line 362-368):
   - Is this "what can you do?" → NO
   - Continue...

2. **Check Service Mode** (line 370):
   - `self.service_index != ["-1"]` → FALSE (it's ["-1"])
   - Skip pre-configured block → Go to auto-select

3. **Build LLM Prompt** (line 410):
   ```
   Given 4 services: 'service 0: if it is a chess game..., service 1: update vector database..., service 2: generate code..., service 3: answer question...', which service can answer the following query? The query is 'Predict next move: e4, e5, Nf3'. For instance, if the query is to predict the next chess move, then select service 0...
   ```

4. **LLM Analyzes** (line 413):
   - LLM sees: "predict next move" + "e4, e5, Nf3"
   - LLM thinks: "This is clearly a chess move prediction"
   - LLM returns: `"service 0"`

5. **Parse Response** (line 416):
   - Regex finds: "service 0"
   - Result: `["0"]`

6. **Filter Valid Services** (line 420):
   - `{"0": "if it is a chess game, predict the next chess move..."}`

7. **Parse Chess Query** (line 432):
   - Detect move sequence pattern: "e4, e5, Nf3"
   - Result: Service 0.1 (sequence of moves)
   - Update: `{"0.1": "predict next move given a sequence of moves"}`

8. **Return to qa.py**:
   - `selected_services = {"0.1": "..."}`
   - qa.py routes to chess service with moves

9. **Execute**:
   - Stockfish predicts next move
   - Returns: "The next moves are from ['Nc6', 'Bb5', 'Nf6', ...]..."

## Key Differences: Auto vs Pre-Selected

| Aspect | Pre-Selected Service | Auto-Select |
|--------|---------------------|-------------|
| **LLM Call** | ❌ No LLM call for service selection | ✅ LLM analyzes and selects |
| **Service Choice** | Uses configured service directly | LLM chooses best match |
| **Prompt** | No service selection prompt | Full prompt with all 4 services |
| **Performance** | Faster (skips LLM) | Slower (requires LLM call) |
| **Flexibility** | Must use selected service | Can choose any service |

## What I Actually Did

**Nothing special!** The auto-select logic was already there - it was working correctly. The problem was:

❌ **Before**: Even pre-selected services were going through LLM analysis  
✅ **After**: Only auto-select goes through LLM analysis

The fix was simply **adding the check at line 370**:
```python
if self.service_index != ["-1"]:
    # Pre-configured - use directly, skip LLM
    return selected_services, service_info_dict

# Only reaches here if auto-select
# Continue with LLM analysis...
```

This separated the two paths clearly:
1. **Pre-selected path**: Skip LLM, use configured service
2. **Auto-select path**: Use LLM to choose (this was already working!)

## Summary

For "User selects Auto → LLM chooses appropriate service", I **didn't add anything new**. The auto-select logic was already implemented and working. 

What I did was **remove the bug** where pre-selected services were also going through this logic, which was interfering with the auto-select mode. By separating the two paths, auto-select now works as originally intended!
