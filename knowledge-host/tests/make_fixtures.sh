#!/usr/bin/env bash
# Build a tiny demo document collection (txt / md / html) for the smoke test.
#   bash tests/make_fixtures.sh /tmp/kb-fixtures
set -euo pipefail
DIR="${1:-/tmp/kb-fixtures}"
mkdir -p "$DIR"

cat > "$DIR/krebs.md" <<'EOF'
# Citric Acid Cycle

## Overview
The citric acid cycle, also known as the Krebs cycle or the TCA cycle, is a
series of chemical reactions that release stored energy through the oxidation of
acetyl-CoA. It occurs in the mitochondrial matrix.

## Discovery
It is named after Hans Krebs, who described the cycle in 1937.
EOF

cat > "$DIR/lighthouse.txt" <<'EOF'
A lighthouse is a tower with a bright light at the top, located at an important
or dangerous place near water. Lighthouses mark coastlines, dangerous shoals and
reefs, and safe harbour entries, and assist in aerial navigation.
EOF

cat > "$DIR/amiga.html" <<'EOF'
<html><head><title>Amiga</title></head><body>
<h1>Amiga</h1>
<p>The Amiga is a family of personal computers introduced by Commodore in 1985.</p>
<h2>Chipset</h2>
<p>The original chipset featured custom co-processors for graphics and sound,
including Agnus, Denise, and Paula.</p>
</body></html>
EOF

echo "fixtures written to $DIR"
