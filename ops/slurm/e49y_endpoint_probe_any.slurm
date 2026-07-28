#!/usr/bin/env bash
#SBATCH --job-name=e49y-ep-any
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=var/artifacts/e49y-ep-any-%j.out
#SBATCH --error=var/artifacts/e49y-ep-any-%j.err

set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" -c \
  'import json, socket, urllib.request; opener = urllib.request.build_opener(urllib.request.ProxyHandler({})); response = opener.open("http://node302:8770/v1/models", timeout=30); payload = json.loads(response.read().decode("utf-8")); assert payload.get("data"); print("probe_node=" + socket.gethostname()); print("remote_frozen_node302_endpoint=ok")'
