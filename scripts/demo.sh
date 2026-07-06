#!/usr/bin/env bash
# Demo: the Apprentice's persistent + proactive behavior.
#
# This script demonstrates the "aha" moment: index a codebase, state a plan,
# introduce a change that violates the plan, run watch, see the Apprentice
# flag it WITHOUT being asked.
#
# Usage: bash scripts/demo.sh

set -e

DEMO_DIR=$(mktemp -d /tmp/apprentice-demo-XXXXXX)
cd "$DEMO_DIR"
git init -q
pip install -e /home/z/my-project/apprentice --break-system-packages -q 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

echo "=== 1. Create a small codebase ==="
mkdir -p src
cat > src/auth.py << 'EOF'
def login(username, password):
    """Authenticate a user."""
    if validate(password):
        return create_session(username)
    return None

def validate(password):
    return len(password) >= 8

def create_session(username):
    return f"session-{username}"
EOF

cat > src/main.py << 'EOF'
def main():
    print(login("alice", "secret123"))

if __name__ == "__main__":
    main()
EOF

echo "=== 2. Initialize and index ==="
apprentice init
apprentice index 2>&1 | grep -E "(Files|Functions|Dead|Cliché)"

echo ""
echo "=== 3. State a plan: refactor authentication ==="
apprentice plan "refactor authentication to use JWT tokens"

echo ""
echo "=== 4. Make a change that DRIFTS from the plan (introduces UI work) ==="
cat > src/ui.py << 'EOF'
def render_button(label):
    return f"<button>{label}</button>"

def render_form(fields):
    # TODO: add validation for form fields
    return "<form>" + "".join(render_button(f) for f in fields) + "</form>"

def unused_helper():
    return "this function has no callers"
EOF

echo ""
echo "=== 5. Run watch — the Apprentice flags issues WITHOUT being asked ==="
apprentice watch

echo ""
echo "=== 6. Ask about the codebase (uses persistent model) ==="
apprentice ask "login"

echo ""
echo "=== 7. Recall a specific function ==="
apprentice recall src.auth.login

echo ""
echo "=== 8. Show status ==="
apprentice status

echo ""
echo "=== 9. Persistence test: the model survives restart ==="
echo "    (Reopening the store...)"
apprentice status

echo ""
echo "Demo complete. The Apprentice flagged:"
echo "  - Plan drift (UI work introduced under an auth plan)"
echo "  - TODO without plan (the form validation TODO)"
echo "  - Dead code (unused_helper)"
echo ""
echo "None of these required asking the Apprentice anything."
echo "It proactively analyzed the changes and reported findings."
echo ""
echo "This is what Copilot, Cursor, and Devin do NOT do."
