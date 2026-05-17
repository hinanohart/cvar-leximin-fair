#!/usr/bin/env bash
# ============================================================================
# release-to-pypi.sh — the ONLY manual step left for cvar-leximin-fair.
# ----------------------------------------------------------------------------
# Everything else (code, tests, CI matrix, GitHub release, version bumps,
# documentation, paper) is automated. PyPI publication is gated on two
# external actions that cannot be done from inside this repo. This script
# walks through both of them.
#
# Time required: ~5 minutes.
# When to run: once, ever. Future releases re-use the same Trusted Publisher.
# ============================================================================

set -euo pipefail

REPO="hinanohart/cvar-leximin-fair"
TAG="${1:-v0.1.1}"
PKG="cvar-leximin-fair"
ENV="pypi"

cat <<'BANNER'

  ╭─────────────────────────────────────────────────────────────────╮
  │ cvar-leximin-fair → PyPI release helper                         │
  │                                                                 │
  │ This script does TWO things:                                    │
  │   1. Walks you through the one-time PyPI Trusted Publisher      │
  │      configuration (manual: a few clicks in a browser).         │
  │   2. Triggers the release workflow on GitHub Actions, which     │
  │      then publishes via OIDC (no API token needed).             │
  ╰─────────────────────────────────────────────────────────────────╯

BANNER

# ----------------------------------------------------------------------------
# Step 1: Trusted Publisher (one-time, manual in browser)
# ----------------------------------------------------------------------------
cat <<EOF
─── Step 1: PyPI Trusted Publisher (one-time) ──────────────────────────────

Open https://pypi.org/manage/account/publishing/ in your browser.

If this is the very first release, the project "$PKG" does not exist on PyPI
yet — that's fine. PyPI lets you add a Trusted Publisher for a project
that does not yet exist; the first successful publish from GitHub creates
the project.

In the "Add a new pending publisher" form, enter exactly:

  PyPI Project Name:    $PKG
  Owner:                hinanohart
  Repository name:      cvar-leximin-fair
  Workflow name:        release.yml
  Environment name:     $ENV

Click "Add". Then return here and press ENTER.

EOF
read -rp "Press ENTER once the Trusted Publisher is configured (or Ctrl-C to abort) ..."

# ----------------------------------------------------------------------------
# Step 2: Confirm tag is in place
# ----------------------------------------------------------------------------
echo
echo "─── Step 2: Confirming git tag '$TAG' exists on remote ────────────────────"
if ! git ls-remote --tags "https://github.com/$REPO.git" "$TAG" | grep -q "$TAG"; then
  echo "ERROR: tag '$TAG' is not on the remote. Push it first:"
  echo "    git tag -a $TAG -m '...'"
  echo "    git push origin $TAG"
  exit 1
fi
echo "OK: tag '$TAG' is on origin."

# ----------------------------------------------------------------------------
# Step 3: Trigger workflow_dispatch
# ----------------------------------------------------------------------------
echo
echo "─── Step 3: Triggering release.yml on GitHub Actions ──────────────────────"
gh workflow run release.yml \
    --repo "$REPO" \
    --ref "$TAG" \
    -f "ref=$TAG"
sleep 3  # let the API register the run

# ----------------------------------------------------------------------------
# Step 4: Watch the run
# ----------------------------------------------------------------------------
echo
echo "─── Step 4: Watching the most-recent release.yml run ──────────────────────"
RUN_ID=$(gh run list \
            --repo "$REPO" \
            --workflow=release.yml \
            --limit 1 \
            --json databaseId \
            -q '.[0].databaseId')
echo "Run ID: $RUN_ID"
gh run watch "$RUN_ID" --repo "$REPO" --exit-status
echo
echo "─── Step 5: Verifying the package is now on PyPI ──────────────────────────"
sleep 5
if pip index versions "$PKG" 2>/dev/null | grep -q .; then
  echo "OK: $PKG is on PyPI."
  echo "    Anyone can now: pip install $PKG"
else
  echo "WARN: pip index could not see $PKG yet (CDN propagation can take a"
  echo "      minute). Try again with: pip index versions $PKG"
fi

cat <<DONE

  ╭─────────────────────────────────────────────────────────────────╮
  │ Done. Future releases:                                          │
  │   1. bump version in pyproject.toml + src/cvar_leximin/__init__ │
  │   2. git tag -a vX.Y.Z -m '...' && git push origin vX.Y.Z       │
  │   3. ./scripts/release-to-pypi.sh vX.Y.Z                        │
  │ Trusted Publisher is sticky — Step 1 above is one-time only.    │
  ╰─────────────────────────────────────────────────────────────────╯
DONE
