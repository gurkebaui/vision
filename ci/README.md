# Continuous integration

`github-actions.yml` is a ready-to-use GitHub Actions workflow (lint + tests on
Python 3.9/3.11/3.12, plus CLI smoke tests).

It is parked here rather than in `.github/workflows/` because the automation
that produced this branch is not authorised to create workflow files. Enable it
with:

```bash
mkdir -p .github/workflows
git mv ci/github-actions.yml .github/workflows/ci.yml
git commit -m "Enable CI"
git push
```

The suite needs no camera and no model bundle, so it runs anywhere.
