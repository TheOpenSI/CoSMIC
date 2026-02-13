# Setup
> [!IMPORTANT]
> Make sure you're in the `app` directory when running these instructions.

## 1. Install [uv](https://docs.astral.sh/uv/getting-started/installation) package and project manager tool.
> [!NOTE]
> For Linux, `uv` installation may differs depends on the distro you are using. Check for which distro are you on first and install based on the recommendation from the distro.

After installing, perform the below commands depends on your OS:
### Windows
```powershell
# Install dependencies from the lock file
uv sync

# Run the backend in FastAPI dev mode (auto reload is enabled by default)
uv run fastapi dev
```

### Linux
```bash
# Install dependencies from the lock file
uv sync

# Run the backend in FastAPI dev mode (auto reload is enabled by default)
uv run fastapi dev
```

The website should be viewable under <ins>**localhost:8000**<ins>
