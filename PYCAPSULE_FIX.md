# PyCapsule Service Error Fix

## Problem

When selecting Service 2 (Code Generation) and asking:
```
"write a python script to print the 100 first prime numbers"
```

You received the error:
```
PyCapsule service encountered an error.
```

## Root Cause

### 1. PyCapsule Container Missing
The **PyCapsule service was not defined** in your `docker-compose.yaml` file. The service was referenced in the code but the Docker container wasn't running.

### 2. Wrong Container Name in Code
The `CodeGenerator` was initialized with default `service_container_name="localhost"`, but should use `"pycapsule"` for Docker networking.

## Solution

### 1. Added PyCapsule Service to docker-compose.yaml

**Added after pipelines service:**
```yaml
pycapsule:
  image: opensicbr/pycapsule:latest
  privileged: true
  container_name: pycapsule
  ports:
    - '8780:8780'
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
    - pycapsule_mount:/app/services/Container/mount_dir/synk_mount
  depends_on:
    - ollama
  networks:
    - cosmic_net
```

**Key features:**
- ✅ Uses port 8780 (PyCapsule default)
- ✅ Has privileged mode (needed for Docker-in-Docker)
- ✅ Mounts Docker socket for code execution in isolated containers
- ✅ Has shared mount volume for code validation
- ✅ Connected to cosmic_net network

### 2. Updated cosmic Service Dependencies

**Changed:**
```yaml
depends_on:
  - ollama
  - pycapsule  # Added this
```

Now cosmic waits for pycapsule to be ready before starting.

### 3. Added pycapsule_mount Volume

**Changed volumes section:**
```yaml
volumes:
  ollama: {}
  open-webui: {}
  pipelines: {}
  volume_configs: {}
  pycapsule_mount: {}  # Added this
```

### 4. Fixed CodeGenerator Initialization

**File:** `src/opensi_cosmic.py` (line 107)

**Before:**
```python
self.code_generator = CodeGenerator()  # Defaults to "localhost"
```

**After:**
```python
self.code_generator = CodeGenerator(service_container_name="pycapsule")
```

Now it correctly points to the PyCapsule container in Docker network.

## How It Works Now

### Request Flow:
```
User Query: "Write Python script to print 100 primes"
    ↓
Service Selection → Service 2 (Code Generation)
    ↓
qa.py routes to code_generator
    ↓
CodeGenerator sends HTTP POST to http://pycapsule:8780/query
    ↓
PyCapsule Container:
  - Receives query
  - Uses qwen2.5-coder model
  - Generates code
  - Tests code in isolated Docker container
  - Debugs if errors found
  - Returns validated code
    ↓
Response: "code generation and validation successful, 
          number of debugging attempt made: X"
    ↓
User receives tested, working code
```

## Next Steps

### 1. Restart Docker Services

Since you already ran `docker-compose up --build -d`, you need to restart to pick up the new configuration:

```powershell
cd F:\cosmic-dev\CoSMIC

# Stop all containers
docker-compose down

# Pull the PyCapsule image (if not already available)
docker pull opensicbr/pycapsule:latest

# Start with new configuration
docker-compose up -d
```

### 2. Verify PyCapsule is Running

```powershell
# Check if pycapsule container is running
docker ps | findstr pycapsule

# Check pycapsule logs
docker logs pycapsule

# Test pycapsule health endpoint
curl http://localhost:8780/health
```

### 3. Test Code Generation

Now try your query again:
```
"write a python script to print the 100 first prime numbers"
```

**Expected Response:**
```
code generation and validation successful, number of debugging attempt made: 1

def get_primes(n):
    ...
```

## Troubleshooting

### If PyCapsule Still Fails:

1. **Check if container is running:**
   ```powershell
   docker ps -a | findstr pycapsule
   ```

2. **Check pycapsule logs for errors:**
   ```powershell
   docker logs pycapsule --tail 50
   ```

3. **Verify network connectivity:**
   ```powershell
   docker exec cosmic ping pycapsule
   ```

4. **Check if port 8780 is accessible:**
   ```powershell
   docker exec cosmic curl http://pycapsule:8780/health
   ```

### Common Issues:

| Issue | Solution |
|-------|----------|
| Image not found | Run `docker pull opensicbr/pycapsule:latest` |
| Port 8780 in use | Change port mapping in docker-compose.yaml |
| Permission denied | Ensure Docker socket is mounted correctly |
| Network error | Verify both services are on cosmic_net |

## Files Modified

1. **`docker-compose.yaml`**
   - Added pycapsule service definition
   - Added pycapsule to cosmic dependencies
   - Added pycapsule_mount volume

2. **`src/opensi_cosmic.py`** (line 107)
   - Changed `CodeGenerator()` to `CodeGenerator(service_container_name="pycapsule")`

## For Local Development (Without Docker)

If you want to run PyCapsule locally outside Docker:

1. Install PyCapsule separately
2. Run it on port 8780
3. Change initialization back to:
   ```python
   self.code_generator = CodeGenerator(service_container_name="localhost")
   ```

## Additional Notes

- PyCapsule requires **privileged mode** because it runs code in isolated Docker containers (Docker-in-Docker)
- The `/var/run/docker.sock` mount allows PyCapsule to create containers for code execution
- Each code execution happens in a fresh, isolated environment for security
- PyCapsule automatically handles debugging and multiple validation attempts

## Verification Checklist

After restart, verify:
- [ ] `docker ps` shows pycapsule container running
- [ ] `docker logs pycapsule` shows no errors
- [ ] Port 8780 is listening
- [ ] Code generation query returns validated code
- [ ] Response includes "number of debugging attempt made: X"
