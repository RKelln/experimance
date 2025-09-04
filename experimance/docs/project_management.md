# Project Management

The Experimance system supports multiple projects (e.g., "experimance", "fire") that can share the same codebase while having project-specific configurations, schemas, and constants.

## Automatic Project Detection

Services automatically detect which project to use through the following priority:

1. **Environment Variable**: `PROJECT_ENV` (highest priority)
2. **Project File**: `projects/.project` file containing the project name  
3. **Default**: "experimance" (fallback)

The detection happens once during `experimance_common` module import, ensuring all services use the same project consistently.

## Setting the Current Project

### Quick Method (Recommended)
```bash
# Set project to "fire"
scripts/project fire

# Set project to "experimance"  
scripts/project experimance

# Check current project
scripts/project
```

**Benefits:**
- ✅ No need to export `PROJECT_ENV` in each terminal
- ✅ Setting persists across terminal sessions
- ✅ Simple, memorable commands
- ✅ Shows available projects when checking status

### Alternative: UV Command
```bash
# Using uv run command (same functionality)
uv run set-project fire
uv run set-project experimance
```

### Manual Method
```bash
# Create/update the .project file manually
echo "fire" > projects/.project

# Or use the Python script directly
uv run python scripts/set_project.py fire
```

### Environment Override
```bash
# Override for current terminal session
export PROJECT_ENV=fire

# Override for single command
PROJECT_ENV=fire uv run -m experimance_core
```

## Project Structure

```
projects/
├── .project              # Current project indicator file (auto-managed)
├── experimance/          # Experimance project config
│   ├── config.toml
│   ├── constants.py
│   └── schemas.py
└── fire/                 # Fire project config
    ├── config.toml
    ├── constants.py
    └── schemas.py
```

## Development Workflow

1. **Set your project once**: `scripts/project fire`
2. **Run services normally**: `uv run -m experimance_core`
3. **Services automatically use the "fire" project configuration**
4. **Switch projects anytime**: `scripts/project experimance`

### Service Logging
Services clearly show which project they're using:
```
2025-09-04 11:22:24,989 - HEALTH - experimance_common.cli - INFO - Using project: fire
2025-09-04 11:22:24,989 - HEALTH - experimance_common.config - INFO - Loaded configuration from projects/fire/health.toml
```

## Technical Implementation

The project detection is handled by `experimance_common.project_utils` with these functions:

- **`detect_project_name()`** - Detects project from environment or file
- **`ensure_project_env_set()`** - Sets PROJECT_ENV if not already set  
- **`set_project()`** - Updates the `.project` file

Detection happens during `experimance_common` module import, before any service code runs.

## Benefits

- **No manual environment exports**: No need to export `PROJECT_ENV` in each terminal
- **Persistent settings**: Project setting persists across terminal sessions  
- **Clear logging**: Services show which project is being used in startup logs
- **Environment override capability**: Can still override with environment variables when needed
- **Graceful fallback**: Defaults to "experimance" if no project is set
- **Centralized management**: All project logic consolidated in `experimance_common.project_utils`
