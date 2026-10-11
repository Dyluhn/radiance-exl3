#!/bin/sh
# upstream_drift.sh -- do the radiance files architectures/qwen4exp_exl3/ was copied from still hold, at
# the pinned release, the bytes architectures/qwen4exp_exl3/UPSTREAM.md records?
#
# USAGE: tests/upstream_drift.sh [REF]
#   REF  a radiance tag to check against (default: the pin, "v<version>" from RADIANCE_VERSION).
#        Checking the pin also checks that the tag still names the pinned commit.
#        Any other REF is the question "would moving to REF need a port?" -- and
#        `tests/upstream_drift.sh v1.0.8` is the negative control: those files changed since, so it
#        must FAIL.
#
# EXIT: 0 PASS -- every recorded file is byte-identical at REF
#       1 FAIL -- a file changed (or the tag moved): port the printed diff, then update UPSTREAM.md
#       2 INFRA -- the answer could not be computed (no network, no tag, a bad RADIANCE_VERSION);
#                  never reported as a pass
#
# ENV: RADIANCE_GIT_URL  where to fetch from: a URL or a local radiance clone
#                        (default https://codeberg.org/StillDeadcode/radiance.git)
#      DRIFT_TIMEOUT     seconds the fetch may take (default 120)
#      TMPDIR            where the throwaway shallow clone goes (about 200 KB; removed on exit)
set -u

here=$(cd "$(dirname "$0")/.." && pwd) || exit 2
upstream_md=$here/architectures/qwen4exp_exl3/UPSTREAM.md
url=${RADIANCE_GIT_URL:-https://codeberg.org/StillDeadcode/radiance.git}

infra() { echo "upstream_drift: INFRA: $*" >&2; exit 2; }

[ -r "$here/RADIANCE_VERSION" ] || infra "no RADIANCE_VERSION at $here (the pin: '<version> <commit>')"
read -r pin_ver pin_commit < "$here/RADIANCE_VERSION" || infra "RADIANCE_VERSION is empty"
[ -n "${pin_commit:-}" ] || infra "RADIANCE_VERSION must be '<version> <commit>', got '$pin_ver'"
ref=${1:-v$pin_ver}

# The recorded hashes: the fenced ```upstream-sha256 block of UPSTREAM.md, one "sha256  path" a line.
[ -r "$upstream_md" ] || infra "no $upstream_md"
pins=$(awk '/^```upstream-sha256/ {on = 1; next} /^```/ {on = 0} on && NF == 2' "$upstream_md")
[ -n "$pins" ] || infra "$upstream_md has no upstream-sha256 block"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/upstream_drift.XXXXXX") || infra "mktemp"
trap 'rm -rf "$tmp"' EXIT INT TERM
# Shallow and blobless: the tag's commit and tree, and only the blobs `git show` asks for below.
# Never prompts (a repository that does not exist answers 401, and git would ask for a password)
# and gives up after DRIFT_TIMEOUT seconds (default 120), so an unreachable host is INFRA, not a hang.
export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true SSH_ASKPASS=true
if ! timeout "${DRIFT_TIMEOUT:-120}" git clone -q --bare --depth 1 --filter=blob:none --branch "$ref" \
        "$url" "$tmp/r" 2> "$tmp/err"; then
    infra "cannot fetch '$ref' from $url: $(tr '\n' ' ' < "$tmp/err")"
fi
commit=$(git -C "$tmp/r" rev-parse --verify --quiet "$ref^{commit}") || infra "'$ref' is not a commit at $url"

echo "upstream_drift: $ref = $commit ($url); pinned $pin_ver = $pin_commit"
status=0
if [ "$ref" = "v$pin_ver" ] && [ "$commit" != "$pin_commit" ]; then
    echo "FAIL  tag $ref now names $commit, and RADIANCE_VERSION pins $pin_commit"
    status=1
fi
echo "$pins" | {
    bad=0
    while read -r want path; do
        have=$(git -C "$tmp/r" show "$commit:$path" 2> /dev/null | sha256sum | cut -d' ' -f1)
        if ! git -C "$tmp/r" cat-file -e "$commit:$path" 2> /dev/null; then
            echo "FAIL  $path does not exist at $ref"
            bad=1
        elif [ "$have" = "$want" ]; then
            echo "PASS  $path  $have"
        else
            echo "FAIL  $path changed: $want at the pin, $have at $ref"
            echo "      port it: git -C <radiance clone> diff $pin_commit $commit -- $path"
            bad=1
        fi
    done
    exit $bad
} || status=1

if [ $status = 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
exit $status
