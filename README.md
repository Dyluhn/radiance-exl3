# radiance-exl3 (working name)

EXL3 routed experts for the radiance inference engine, as out-of-tree plugins: an `exl3` quantiser that imports
packed EXL3 planes, a host reference kernel library (`exl3ref`), a gfx12 device kernel library (`exl3r4d`) and a
`("qwen4exp", "exl3")` architecture plugin. Nothing in radiance is modified.

Pinned engine: `RADIANCE_VERSION` (release tag + commit). Plan and evidence: see `notes/` and `NOTES.md`.
