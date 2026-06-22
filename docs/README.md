# qminweight documentation

This directory is the documentation site for **qminweight**. The pages are plain Markdown, so
they render directly on GitHub and PyPI; the `mkdocs.yml` here additionally wires them into
a browsable site with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme.

The Markdown pages live in `pages/` (the MkDocs `docs_dir`); `mkdocs.yml` and this README
sit alongside it.

## Pages

| File | Contents |
|---|---|
| `pages/index.md` | What qminweight is, the headline results, the feature list |
| `pages/installation.md` | Building the native library and installing the package |
| `pages/quickstart.md` | Minimal Python and CLI examples |
| `pages/api.md` | Python API reference (`css_distance`, `classical_distance`, `Result`, `qminweight.codes`) |
| `pages/cli.md` | The `qminweight` / `python -m qminweight` command line |
| `pages/algorithms.md` | BZ, connected cluster, MITM — how they work and when to use each |
| `pages/benchmarks.md` | Measured numbers vs the reference, and how to reproduce |
| `pages/contributing.md` | Repo layout, building, running the tests, conventions |

## Building the site locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open <http://127.0.0.1:8000>. `mkdocs serve` live-reloads as you edit.

The MkDocs commands must be run **from inside this `docs/` directory** (where `mkdocs.yml`
lives):

```bash
cd docs
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build            # render a static site into ../site/
```

`mkdocs build` writes the static HTML to a `site/` directory at the repository root (per
the `site_dir` setting in `mkdocs.yml`).
