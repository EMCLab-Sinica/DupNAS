set -euo pipefail

grep -rl 'REPO_ROOT' stm_projects | xargs sed -i "s#REPO_ROOT#$PWD#g"