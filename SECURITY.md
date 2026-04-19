# Security Policy

## Supported versions

`bayes-hdc` is in alpha. Only the latest commit on `main` and the most
recent tagged release receive security fixes.

| Version | Supported |
|---------|-----------|
| `main` (rolling) | ✅ |
| latest tagged release | ✅ |
| older tagged releases | ❌ |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Use one of the following private channels:

1. **GitHub Security Advisories** — preferred. Open a draft advisory at
   <https://github.com/rlogger/bayes-hdc/security/advisories/new>.
2. **Email** — send a description and minimal reproduction to
   `rajdeeps@usc.edu` with subject line `[bayes-hdc security]`.

Please include:

- A description of the issue and the conditions that trigger it.
- A minimal reproduction (Python snippet is ideal).
- The version or commit SHA you observed it on.
- The JAX / jaxlib versions, platform, and accelerator.
- Any suggested fix or mitigation.

## Response timeline

- **Acknowledgement** within 72 hours.
- **Initial assessment** within 7 days.
- **Patch or mitigation** coordinated with the reporter; released as a
  new tag with a corresponding GitHub Security Advisory.

## Scope

In scope:

- Code execution, sandbox escape, or privilege escalation in `bayes_hdc/`.
- Numerical correctness bugs that silently produce wrong hypervectors in
  published operations (these are prioritised as security issues when
  they affect capacity or retrieval guarantees).
- Vulnerabilities in the release / packaging pipeline.

Out of scope:

- Bugs in third-party dependencies (report those upstream).
- Performance regressions.
- Non-security bugs — please file a regular issue instead.
