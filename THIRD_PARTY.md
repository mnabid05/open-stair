# Licensing and provenance

## This repository

Original Python code, documentation, tests, and examples in this repository are licensed
under [Apache-2.0](LICENSE). The copyright notice and research attribution are in [NOTICE](NOTICE).
Apache-2.0 is an [OSI-approved open-source license](https://opensource.org/license/apache-2.0).
It permits use, modification, and redistribution, including commercial use, subject to its
terms. Keep the license and applicable notices when redistributing the project.

This is independently written software based on the published STAIR method. No code from
the IBM authors' repository, SearchTome examples, or pretrained model weights is vendored.
The small handbook and its questions are original examples. The research contribution is
credited in [CITATION.cff](CITATION.cff) and [docs/research.md](docs/research.md). IBM's name
describes the method's provenance and does not imply affiliation or endorsement.

## Dependencies and external artifacts

Dependencies are installed separately by the package manager. Their licenses and bundled
third-party notices remain applicable; this repository does not relicense them.

| Direct dependency | Upstream licensing reference |
| --- | --- |
| PyTorch | [PyTorch license and third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| PEFT | [Apache-2.0](https://github.com/huggingface/peft/blob/main/LICENSE) |
| pypdf, optional | [BSD-3-Clause](https://github.com/py-pdf/pypdf/blob/main/LICENSE) |
| Ruff, development | [MIT](https://github.com/astral-sh/ruff/blob/main/LICENSE) |
| build, development | [MIT](https://github.com/pypa/build/blob/main/LICENSE) |
| Twine, release validation | [Apache-2.0](https://github.com/pypa/twine/blob/main/LICENSE) |

The default [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and
[Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model cards identify
their weights as Apache-2.0. Downloaded models remain separate artifacts; consult the exact
model revision and its license when exporting or distributing trained/merged weights.
Alternative models may have different terms. Your input documents and generated datasets
also retain their applicable rights. SearchTome is not included in any release archive.

The release archive contains source only. It excludes local environments, credentials,
corpus artifacts, downloaded/trained models, and package-manager caches.
