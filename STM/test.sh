set -euo pipefail

case "$1" in
    ten)
        for i in {1..10}; do echo "$i"; sleep 1; done
        ;;
    hundred)
        for i in {1..100}; do echo "$i"; sleep 1; done
        ;;
    *)
        echo "unknown target: $1"
        exit 1
        ;;
esac
