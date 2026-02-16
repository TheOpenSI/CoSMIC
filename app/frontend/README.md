# Setup
```bash
# Install dependencies.
bun install

# Start the frontend (using `bunx` to let Vite using Bun runtime instead of Node
# runtime).
bunx --bun vite
```
From here, you have 2 options

## 1. FE work only
- The webiste will be accessible at [localhost:5173](http://localhost:5173/)

## 2. FE + BE work
```bash
# After finish playing/developing around the frontend, make sure to generate the
# compiled version of it so backend can serve.
bunx --bun vite build
```
Move over to [backend](../backend/README.md) directory to complete the rest of
the setup before you can use the new UI platform.
