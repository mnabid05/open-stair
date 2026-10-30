# Publish on GitHub

The repository is prepared as an Apache-2.0 source project. No GitHub repository or PyPI
package is created automatically. Use a new public GitHub repository after reviewing the
source. The source-only ZIP includes the hidden `.github` configuration; extract it before
uploading or committing its contents. Uploading the ZIP alone does not create a usable source
tree in GitHub's Code tab.

## Verify the release

From the repository root with Python 3.12:

```sh
python -m pip install '.[dev,pdf]'
python scripts/release.py --check
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 python examples/smoke_train.py --output artifacts/release-smoke
ruff check .
ruff format --check .
python -m build
python -m twine check dist/open_stair-0.2.1*
python scripts/release.py --archive
```

The release checker examines the explicitly included source files for credential patterns,
machine-specific paths, model artifacts, broken documentation links, and inconsistent
version metadata. It is a targeted check, not a guarantee that arbitrary added content is
safe to publish. The ZIP omits environments, Git metadata, corpora, trained weights, and
local build outputs. Its allowlist is in `scripts/release.py`.

## Create the GitHub repository

Keep all source, `.github`, LICENSE, NOTICE, and citation files. In a freshly extracted ZIP,
initialize Git first with `git init -b main`; this workspace already has a Git repository.
Then, from the project root:

```sh
git add .github .gitattributes .gitignore CHANGELOG.md CITATION.cff CODE_OF_CONDUCT.md CONTRIBUTING.md LICENSE MANIFEST.in NOTICE README.md SECURITY.md THIRD_PARTY.md docs examples pyproject.toml scripts src tests
git diff --cached --stat
git commit -m "Release Open STAIR 0.2.1"
gh repo create open-stair --public --source=. --remote=origin
git push -u origin main
```

Choose a different repository name if `open-stair` is unavailable in your account. These
commands publish under the account authenticated in `gh`; review `gh auth status` first.
If using an existing repository, add its actual URL as `origin` instead of running
`gh repo create`. This release's metadata and installation instructions target
`https://github.com/mnabid05/open-stair`; update them if publishing a fork under another name.

After pushing, wait for the Actions workflow to pass. The workflow validates lightweight
Python compatibility, tests CPU model training on Linux and Windows, builds source/wheel
artifacts, and checks release contents. Those remote platform checks have not run until the
repository is pushed. Enable **Settings → Code security → Private vulnerability reporting**
so SECURITY.md's private reporting route is available.

## Tag a GitHub release

When CI passes for the chosen commit:

```sh
git tag -a v0.2.1 -m "Open STAIR 0.2.1"
git push origin v0.2.1
gh release create v0.2.1 dist/open_stair-0.2.1.tar.gz dist/open_stair-0.2.1-py3-none-any.whl dist/open-stair-0.2.1-github.zip --title "Open STAIR 0.2.1" --notes-file CHANGELOG.md
```

No PyPI name has been reserved or published. Developers can install a clone with
`python -m pip install .`, or install the release wheel directly. A later PyPI release is a
separate action; first confirm name availability, configure trusted publishing, and set the
actual repository/documentation URLs in `pyproject.toml`. Do not claim a PyPI installation
command works before publication.

## Release claims

Describe this as an independent alpha implementation of STAIR's published method with tested
weight training and export. It has not reproduced IBM's full Mistral/SearchTome benchmark.
Do not advertise IBM affiliation, production certification, guaranteed factual accuracy,
or universal hardware compatibility. Code and original examples are Apache-2.0; external
models and data keep their own terms, as explained in [THIRD_PARTY.md](../THIRD_PARTY.md).
