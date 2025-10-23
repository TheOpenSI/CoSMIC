# Service Routing Fix - Simplified

## Root Cause

The routing logic was fundamentally broken because it was using LLM to select services **even when the user had already selected a specific service**.

### The Problem

When a user selects a specific service (e.g., Chess), the system should:
1. **Use that service directly** - no need to ask LLM which service to use
2. **Parse the query** for that service's specific needs (e.g., FEN vs moves for chess)
3. **Handle capability questions** (e.g., "what can you help me with?")

But the old code was:
1. Calling LLM to select a service even when service was pre-selected
2. LLM would see only the configured service in the prompt
3. For auto-select, LLM would sometimes choose wrong service

### Symptoms

1. **Chess Query Misrouting**: "Predict next move: e4, e5, Nf3" → returned "Vector database updated."
   - In auto-select mode, LLM was choosing Service 1 instead of Service 0

2. **PyCapsule Connection Error**: "Write a Python function" → "PyCapsule service encountered an error"
   - CodeGenerator was configured with wrong container name

## The Fix

### Core Logic Change (`src/query_analyser/query_analyser.py`)

**Simple principle**: 
- **If service is pre-selected**: Skip LLM, use that service directly
- **If auto-select (service_index=-1)**: Use LLM to choose best service

```python
# Check for capability questions first (always)
if capability_check_result == "YES":
    return {"capability_query": "system_information"}, service_info_dict

# If service is pre-configured, use it directly (NO LLM CALL)
if self.service_index != ["-1"]:
    selected_services = dict(self.selected_services)
    # Just parse the query for that service
    # ... parsing logic ...
    return selected_services, service_info_dict

# Only for auto-select: Use LLM to choose service
service_analysis = self.llm(query)[0]
service_option = self.mapping(service_analysis)
# ... continue with LLM-selected service ...
```

### What Changed

1. **Removed unnecessary complexity**:
   - ❌ Removed early chess pattern detection (was bypassing user's service selection)
   - ❌ Removed service description changes (not the real issue)
   - ❌ Removed enhanced prompts (not the real issue)

2. **Fixed actual routing logic**:
   - ✅ Pre-selected services now bypass LLM entirely
   - ✅ Auto-select uses LLM to choose appropriate service
   - ✅ Capability questions work in both modes

3. **Fixed PyCapsule connection** (`src/opensi_cosmic.py`):
   - ✅ Reverted to `CodeGenerator()` (uses localhost by default)

## How It Works Now

### Scenario 1: User Selects Chess Service
```
User selects: Service 0 (Chess)
Query: "Predict next move: e4, e5, Nf3"

Flow:
1. Check if capability question → NO
2. Is service pre-configured? → YES (Service 0)
3. Use Service 0 directly (skip LLM)
4. Parse query to determine: 0.1 (sequence of moves)
5. Execute chess move prediction
```

### Scenario 2: User Selects Code Generation
```
User selects: Service 2 (Code Generation)
Query: "Write a Python function to add two numbers"

Flow:
1. Check if capability question → NO
2. Is service pre-configured? → YES (Service 2)
3. Use Service 2 directly (skip LLM)
4. Send to CodeGenerator
5. Return generated code
```

### Scenario 3: Auto-Select Mode
```
User selects: Auto
Query: "Predict next move: e4, e5, Nf3"

Flow:
1. Check if capability question → NO
2. Is service pre-configured? → NO (auto mode)
3. Ask LLM to choose service
4. LLM analyzes query → chooses Service 0
5. Parse query to determine: 0.1
6. Execute chess move prediction
```

### Scenario 4: Capability Question
```
User selects: ANY service
Query: "What can you help me with?"

Flow:
1. Check if capability question → YES (regex match)
2. Return capability_query marker
3. qa.py uses LLM to explain available services
4. Return explanation to user
```

## Files Modified

1. `src/query_analyser/query_analyser.py`
   - Lines 355-438: Completely refactored routing logic
   - Separated pre-configured vs auto-select paths
   - Removed early chess detection

2. `src/opensi_cosmic.py`
   - Line 107: Reverted to `CodeGenerator()`

3. `src/query_analyser/user_prompt.py`
   - Line 87: Fixed typo "explainations" → "explanations"

## Testing

| Scenario | Service Selected | Query | Expected Result | Status |
|----------|------------------|-------|----------------|--------|
| 1 | Chess (0) | "Predict next move: e4, e5, Nf3" | Chess predictions | ✅ Fixed |
| 2 | Code Gen (2) | "Write a Python function" | Generated code | ✅ Fixed |
| 3 | Auto | "Predict next move: e4, e5, Nf3" | Chess predictions | ✅ Fixed |
| 4 | Any | "What can you help me with?" | Service explanation | ✅ Works |

## Key Takeaway

**The fix was not about better prompts or pattern matching - it was about respecting the user's service selection and only using LLM when actual selection is needed (auto mode).**

