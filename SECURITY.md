# Security policy

## Reporting

Use this repository's **Security → Report a vulnerability** feature for sensitive reports
when available. If private reporting is not enabled, open an issue requesting a private
security contact without including exploit details, credentials, or private documents.
Maintainers should enable private vulnerability reporting when publishing the repository.

Include the affected release, a minimal reproduction, impact, and a proposed fix if known.
Only the latest 0.2.x release receives fixes; the software is currently alpha.

## Trust boundaries

- Use trusted model artifacts. The loaders require safetensors and disable remote Python
  code, but they do not make model weights or third-party dependencies risk-free.
- Corpus snapshots, generated question files, and learned weights can retain source data.
  Review them before sharing; model export does not anonymize training data.
- The ToC and question are untrusted text passed to a model. This library returns source
  sections and does not execute generated commands or provide agent tools.
- Parsing arbitrary PDF files should happen in an appropriately isolated environment.
- The library has no authentication or network service. Applications exposing it remotely
  must implement their own authentication, input limits, and resource management.

Do not include secrets, private corpora, or trained models in bug reports or pull requests.
