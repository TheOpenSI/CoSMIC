> [!WARNING]
> This directory is an attempt to stop relying on [OpenWebUI](https://github.com/open-webui/open-webui) and create our own UI platform after having continuous issues from catching up with their new changes. Things will DEFINITELY BREAK, TEAR APART, BURN DOWN TO THE ULTERNESS OF VORTEX!! If you're okay with it, simply ignore this warning.

# Setup
## 1. Clone repository
```bash
git clone --single-branch -b bing-dev https://github.com/TheOpenSI/CoSMIC.git   # HTTPS cloning
git clone --single-branch -b bing-dev git@github.com:TheOpenSI/CoSMIC.git       # SSH cloning
```

## 2. Run setup (depends on your usage)

### 2.1 Demonstration (TBA)
### 2.2 Development
Depends on which part of the codebase you wanted to contribute, here is the preferred way to go for:
> [!NOTE]
> Due to how the head of this project wanted to setup the development, FE works are done completely from a sepearate GitHub repo. Therefore, the docs I've here (for now) will soon be merged over to that repo. In that case, this note will be deleted.
1. **FE developmenent**: take a look at the documentation in [here](FE.md) (for now). After the merged is done, please take a look at [frontend](frontend/README.md) directory.
2. **BE developmenent**: take a look at the documentation in [backend](backend/README.md) directory.

# Convention
Choose the section that you needed to see and start from there
1. [Git](doc/GIT.md)
2. [Frontend](doc/REACT.md)
3. [Backend](doc/PYTHON.md)
4. [Database](doc/POSTGRES.md)
