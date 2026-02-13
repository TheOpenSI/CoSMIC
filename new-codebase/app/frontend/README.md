# Setup
> [!IMPORTANT]
> Make sure you're in the `app` directory when running these instructions.

Perform the below commands depends on your OS:
## Windows
```powershell
# Go into specified directory
Set-Location -Path ".\frontend"

# Make sure to run this. Otherwise, the platform cannot be build let alone accessable
bun run build

# Run this command if you'd like to test/work on the FE only
# bun run dev
```

## Linux
```bash
# Go into specified directory
cd "./frontend"

# Make sure to run this. Otherwise, the platform cannot be build let alone accessable
bun run build

# Run this command if you'd like to test/work on the FE only
# bun run dev
```

You should be able to see a new `dist` directory within the specified directory. Check by running:
## Windows
```powershell
Get-Directory -Path ".\frontend"
```

## Linux
```bash
ls -lA --color=auto ".\frontend"
```

Then, moving on to setting up the [backend](../backend/README.md).
