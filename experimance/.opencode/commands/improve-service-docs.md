---
name: improve-service-docs
description: Consolidate and improve docs for a service
---

You are improving documentation for a service at @services/<service>/.
If a service name is provided, use @services/$ARGUMENTS/.

Policy:
1) Root files: keep only ALL-CAPS well-known files at the service root (README.md, CHANGELOG.md, LICENSE). Everything else should live under `services/<service>/docs/` with lowercase filenames.
2) Consolidate: merge existing doc content (root and src subfolders) into `docs/*.md`, removing deprecated or redundant files after migration.
3) README.md: update links and add a short "Additional Docs" index with one-line descriptions per doc.
4) Add `docs/index.md` that lists all service docs with one-line descriptions.
5) Include a lightweight `docs/roadmap.md` for the service with near-term goals and known gaps.
6) Preserve meaning: do not change technical behavior or commands; only edit for clarity, structure, and consistency.
7) Avoid drift: remove old docs once consolidated and ensure all references point to new paths.
8) GitHub browsability: every source package directory under `src/` (e.g. `src/my_package/`) must have a README.md. GitHub renders these inline when browsing the folder. Keep them brief — one-paragraph description, quick-start commands, and a link to the full doc in `docs/`. Do not duplicate content from `docs/`. Check all packages exist and add any that are missing.

Recommendations for better docs:
- Write for the fastest path to success: lead with “What this service does” and “Quick start.”
- Keep commands copy-pasteable; avoid stale references and duplicates.
- Use consistent section order: Overview → Setup → Configuration → Usage → Testing → Troubleshooting → Integrations.
- Prefer small, focused docs over long, mixed-purpose files.
- Include environment assumptions (OS, hardware, required services) near the top.
- Use stable, repo-relative links; avoid external URLs unless essential.
- Add “When to use / When not to use” for optional features.
- If a doc describes a subsystem, include a “Files touched” list.

Docs vs code sanity check:
- Verify commands, flags, and paths exist in the repo (match code/CLI help if available).
- Validate config keys and sections against config models or sample config files.
- Spot-check any referenced scripts and test files exist and are named correctly.
- Confirm message types/event names match the schemas and enums in code.

Optional history check:
- Review recent service commits (including docs) to catch drift and recent behavior changes.

Output:
- List of new/updated doc files
- Removed/renamed files
- Any open questions or conflicts

Validation:
- Run `uv run python scripts/check_docs_links.py` and report results
