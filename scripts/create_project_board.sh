#!/usr/bin/env bash
set -euo pipefail

# Create the MineCombat Projects v2 board and add all open issues.
#
# GitHub Projects requires the `project` and `read:project` scopes. If the
# command below fails, run:
#
#   gh auth refresh -s project,read:project
#
# Then rerun this script.

REPO="${REPO:-MAS-W-415/MineCombat}"
OWNER="${OWNER:-${REPO%%/*}}"
TITLE="${TITLE:-MineCombat Roadmap}"

if ! gh auth status -t 2>&1 | grep -q "project"; then
  echo "Missing GitHub project scope." >&2
  echo "Run: gh auth refresh -s project,read:project" >&2
  exit 1
fi

NUMBER="$(gh project list --owner "$OWNER" --format json \
  --jq ".projects[] | select(.title == \"$TITLE\") | .number" | head -n1)"

if [[ -z "$NUMBER" ]]; then
  echo "Creating project '$TITLE'..."
  NUMBER="$(gh project create --owner "$OWNER" --title "$TITLE" --format json --jq '.number')"
else
  echo "Reusing project '$TITLE' (#$NUMBER)."
fi

field_exists() {
  gh project field-list "$NUMBER" --owner "$OWNER" --format json \
    --jq '.fields[].name' | grep -qx "$1"
}

if ! field_exists "Stage"; then
  gh project field-create "$NUMBER" --owner "$OWNER" \
    --name "Stage" --data-type SINGLE_SELECT \
    --single-select-options "M0,M1,M2,M3,M4" >/dev/null
fi

if ! field_exists "Area"; then
  gh project field-create "$NUMBER" --owner "$OWNER" \
    --name "Area" --data-type SINGLE_SELECT \
    --single-select-options "env,reward,train,eval,infra,docs" >/dev/null
fi

if ! field_exists "Priority"; then
  gh project field-create "$NUMBER" --owner "$OWNER" \
    --name "Priority" --data-type SINGLE_SELECT \
    --single-select-options "P0,P1,P2" >/dev/null
fi

while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  gh project item-add "$NUMBER" --owner "$OWNER" --url "$url" >/dev/null
done < <(gh issue list --repo "$REPO" --state open --limit 100 --json url --jq '.[].url')

echo "Project board ready: https://github.com/users/$OWNER/projects/$NUMBER"
