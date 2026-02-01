#!/bin/bash

# --- Default Config ---
LOCAL_PORT=8873
REMOTE_PORT=873
RSYNC_PASSWORD="swe_bench_sync"
export RSYNC_PASSWORD

# --- Help Menu ---
usage() {
  echo "Usage: $0 -s <source_path> -d <dest_url>"
  echo "  -s : Local source directory (e.g., ./swe-bench/)"
  echo "  -d : Destination path inside rsync container (e.g., /data/project1)"
  exit 1
}

# --- Parse Arguments ---
while getopts "s:d:" opt; do
  case $opt in
    s) SOURCE_PATH="$OPTARG" ;;
    d) DEST_URL_PATH="$OPTARG" ;;
    *) usage ;;
  esac
done

# Check if arguments are provided
if [[ -z "$SOURCE_PATH" || -z "$DEST_URL_PATH" ]]; then
  usage
fi

# Construct the full rsync URL
FULL_DEST="rsync://root@localhost:$LOCAL_PORT$DEST_URL_PATH"

echo "------------------------------------------------"
echo "Initializing 90GB resilient transfer..."
echo "Source: $SOURCE_PATH"
echo "Target: $FULL_DEST"
echo "------------------------------------------------"

# --- The Magic Loop ---
while true; do
  # 1. Start Port-Forward if not running
  if ! lsof -i :$LOCAL_PORT > /dev/null; then
    echo "[$(date +%T)] Opening tunnel via kubectl..."
    kubectl port-forward svc/rsync-service $LOCAL_PORT:$REMOTE_PORT > /dev/null 2>&1 &
    PF_PID=$!
    sleep 3 
  fi

  # 2. Run rsync with total progress bar
  # We use --info=progress2 for a cleaner UI on large transfers
  rsync -avzP --info=progress2 --timeout=30 --inplace "$SOURCE_PATH" "$FULL_DEST"

  # 3. Handle Exit Codes
  RESULT=$?
  if [ $RESULT -eq 0 ]; then
    echo "------------------------------------------------"
    echo "SUCCESS: Sync Complete!"
    kill $PF_PID 2>/dev/null
    exit 0
  elif [ $RESULT -eq 20 ] || [ $RESULT -eq 30 ]; then
    echo "[$(date +%T)] Connection timeout/interrupted. Restarting tunnel..."
  else
    echo "[$(date +%T)] Rsync error code $RESULT. Retrying in 5s..."
  fi

  kill $PF_PID 2>/dev/null
  sleep 5
done