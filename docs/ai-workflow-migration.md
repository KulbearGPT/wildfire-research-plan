# AI workflow migration record

Assessed on 2026-09-22. This records the migration, not instructions to reread
before every task. The maintained agreement is [AGENTS.md](../AGENTS.md), with
operational details in [DEVELOPMENT.md](DEVELOPMENT.md).

## Decisions and instruction audit

The user approved a short project agreement, two narrowly triggered skills,
reuse of existing runners, proportional checks, and coherent commits. The goal
is faster research iteration with intact scientific and resource boundaries.

| Observed source | Decision |
|---|---|
| No root `AGENTS.md` or repository skills before this change | Add one short agreement and two skills in `.agents/skills/` |
| Current reproduction documents and `scripts/research/` already define execution | Reference them; do not introduce another experiment framework |
| `docs/superpowers/` contains earlier plans with workflow directives | Preserve historical records; exclude those directives from new-task defaults |
| Session exposes Superpowers 6.4.1, including broad automatic skill loading and staged design approvals | Replace that default workflow for this project with the user-approved agreement |
| User Codex config already selects `gpt-6-astra` with `low` reasoning | Preserve these defaults in portable project config; do not force maximum effort |
| CLI 0.147.0 local plugin listing reports no installed marketplace plugins | Do not mistake the remote session's skill catalog for local marketplace configuration |
| No `AGENTS.md` or `AGENTS.override.md` found at the checked `/scratch`, `/scratch/kulbear`, or user Codex directory | No existing files at those locations needed replacement; this is not proof that all host instructions are absent |

The supplied `CV_Graphics_Fast_Prototype_AGENTS_v3` template contributed the
minimal evidence cycle, fair comparison, failure classification, budget limits,
and Slurm provenance requirements. General geometry/camera rules and its full
project/experiment forms were not copied into this wildfire repository.
Existing notes and machine-generated artifacts remain the experiment record.

## Configuration scope and activation

[.codex/config.toml](../.codex/config.toml) sets the project model and reasoning
defaults. It also disables `superpowers@openai-curated-remote` **when installed
as a local-marketplace plugin**. This is a preventive local setting, not evidence
that the remote plugin in the current conversation has been disabled.

Official configuration guidance distinguishes local-marketplace settings from
workspace-managed plugin state. Project configuration cannot override enforced
host policy or remove instructions already injected into a running conversation.
No user-global configuration, plugin cache, or workspace installation was changed.

Start a fresh Codex task with this repository as its working directory. A
supported client loads project config only for a trusted repository. Confirm the
effective model/effort and that `wildfire-experiment` and `wildfire-run-diagnosis`
appear in its skill catalog. If managed Superpowers skills still appear, their
availability is controlled by the hosting client/workspace; change that setting
there if desired. Do not claim that editing `AGENTS.md` uninstalled a plugin.

The project agreement asks for task-matched skill use, completion of authorized
work, and no routine approval ceremony. When a higher-priority host instruction
still imposes a conflicting workflow, report the exact source and limitation.
Do not silently bypass it or modify shared settings for other projects.

## Acceptance cases

These are small workflow checks, not scientific experiments or model benchmarks.

| Request | Expected behavior |
|---|---|
| Update an English onboarding link | Edit English source, render the affected course pages, inspect links; no research skill or GPU run |
| Diagnose a pending/failed job from supplied evidence | Use the diagnosis skill, distinguish queue/exit/artifact state, preserve evidence; no unrequested cancellation or resubmission |
| Prepare one matched method/control pilot without running it | Use the experiment skill, identify protocol and missing budget/input facts, produce a reviewable command/config; no job submission |

Static validation and acceptance review are recorded below as they are completed.
No improvement in token cost, latency, or scientific performance is claimed from
writing these instructions. A future real task is the appropriate place to check
whether the new defaults reduce unnecessary reading and interruptions.

## Sources

Official guidance checked on 2026-09-22:

- [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra): narrow skill triggers, read by task, reduce rigid recipes, and define completion.
- [GPT-6 model guidance](https://developers.openai.com/api/docs/guides/latest-model): autonomy, sensitivity to instruction files, and proportional verification.
- [Build skills](https://learn.chatgpt.com/docs/build-skills): repository discovery under `.agents/skills/` and concise descriptions.
- [Config basics](https://learn.chatgpt.com/docs/config-file/config-basic): trusted-project configuration and precedence.
- [Plugin configuration](https://developers.openai.com/plugins/build/plugins#enable-or-disable-a-plugin-for-a-repo): local-marketplace enablement does not override workspace-managed plugin state.

The file layout and the choice of two project skills are project decisions,
not requirements imposed by OpenAI.
