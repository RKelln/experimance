# Roadmap

## Near-term goals

- Align service documentation with current generator strategies and config models.
- Consolidate VastAI and model server guidance into stable, service-local docs.
- Document audio generation setup and testing flows more clearly.
- Add focused docs for dynamic generator selection and image-to-image workflows.

## Known gaps

- Some ZMQ troubleshooting scripts still mention the legacy events flow.
- Audio message schemas live in project-specific schemas (e.g., `projects/fire/schemas.py`).
- Several generator docs assume default values without clarifying project-specific overrides.

## Follow-ups

- Audit `validate_zmq_addresses.py` for stale guidance about the unified events channel.
- Add a small doc for audio generation caching and metadata.
