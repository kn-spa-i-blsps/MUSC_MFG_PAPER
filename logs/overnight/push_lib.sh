# Sourced by the pod chains (durability fix 2026-11-02, after the 3rd mid-run balance stop).
# checkpoint MSG PATH...  : git add -f the existing paths, commit if anything is staged, pull --rebase, push (5 tries).
#                           Serialised with flock so two writers never race on .git/index.lock. Never exits the chain.
# validate_run_dirs DIR.. : for every metrics_seed*.json under DIR: json must parse AND logs_seed*.npz must load every
#                           array fully; otherwise both files are deleted so the skip-existing logic re-runs that seed.
checkpoint() {
  local msg=$1; shift
  (
    flock -w 600 9 || { echo "$(date '+%F %T') checkpoint: lock timeout"; exit 0; }
    for f in "$@"; do [ -e "$f" ] && git add -f "$f"; done
    if git diff --cached --quiet; then echo "$(date '+%F %T') checkpoint: nothing new ($msg)"; exit 0; fi
    git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || exit 0
    if [ -n "$NO_PUSH" ]; then echo "$(date '+%F %T') checkpoint committed locally (NO_PUSH) $(git rev-parse --short HEAD): $msg"; exit 0; fi
    for i in 1 2 3 4 5; do
      git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && {
        echo "$(date '+%F %T') checkpoint pushed $(git rev-parse --short HEAD): $msg"; exit 0; }
      echo "$(date '+%F %T') checkpoint push failed (try $i)"; sleep 30
    done
  ) 9>"${MUSC_LOCK:-/workspace/.musc_git.lock}"
}

validate_run_dirs() {
  python - "$@" <<'PY'
import glob, json, os, sys
import numpy as np
ok = bad = 0
for d in sys.argv[1:]:
    for m in sorted(glob.glob(os.path.join(d, "**", "metrics_seed*.json"), recursive=True)):
        z = m.replace("metrics_seed", "logs_seed").replace(".json", ".npz")
        try:
            json.load(open(m))
            with np.load(z) as f:
                for k in f.files:
                    np.asarray(f[k]).sum()
            ok += 1
        except Exception as e:
            bad += 1
            print(f"invalid -> deleted: {m} ({type(e).__name__}: {e})")
            for p in (m, z):
                if os.path.exists(p):
                    os.remove(p)
print(f"validate: {ok} complete seeds kept, {bad} invalid deleted")
PY
}
