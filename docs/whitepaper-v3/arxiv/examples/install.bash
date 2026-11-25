(
  set -euo pipefail
  installer=$(mktemp)
  trap 'rm -f -- "$installer"' EXIT
  curl -q -fsSL --proto '=https' --tlsv1.2 -o "$installer" \
    https://raw.githubusercontent.com/neverhuman/dope/51df4554bc292c3ad59ec2099fccbbde19cffadc/scripts/install.sh
  printf '%s  %s\n' 808729d2382fc37fbdcf69520d808430536b858074387198920b250033baa213 "$installer" |
    sha256sum --check --status
  bash "$installer" --standalone-prefix "$HOME/.local/opt/dope/3.0.0" \
    --tag dope-engine-v3.0.0 --source-digest e15c415bad9554f7c419dd240aa670d1a79a27fe
)
