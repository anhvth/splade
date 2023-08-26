#!/bin/bash

CURDIR="/Users/anhvth/gitprojects/splade/"
REMOTE_USER_HOST="35" # Assuming "35" resolves to your remote host
REMOTE_DIR="/gitprojects/splade/"
CONTROL_SOCKET="/tmp/${REMOTE_USER_HOST}_ssh_mux"

EXCLUDE_LIST=(
    # ".git"
    # ".gitignore"
    ".DS_Store"
    "sync.sh"
    "outputs"
    # "data"
)

sync_func() {
    # Use rsync with the established connection
    rsync -avzhe "ssh -S $CONTROL_SOCKET" --exclude-from=<(for i in "${EXCLUDE_LIST[@]}"; do echo "$i"; done) \
        "$CURDIR/" "${REMOTE_USER_HOST}:${REMOTE_DIR}" --max-size=2m --delete
}

# Establish master SSH connection
if [ ! -e "$CONTROL_SOCKET" ]; then
    echo "Establishing SSH connection to $REMOTE_USER_HOST..."
    ssh -N -f -M -S "$CONTROL_SOCKET" -o ExitOnForwardFailure=yes "$REMOTE_USER_HOST"
fi

# Monitor changes in CURDIR with fswatch and run sync_func upon change
fswatch -o "$CURDIR" | while read -r event; do
    sync_func
done

# Close the master SSH connection after you're done (optional)
ssh -S "$CONTROL_SOCKET" -O exit "$REMOTE_USER_HOST"

