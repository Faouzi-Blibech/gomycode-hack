#!/bin/sh
set -eu

# A host-side Ollama URL commonly uses localhost. Inside a container that name
# means the container itself, so translate only loopback URLs to Docker's host.
host_url() {
  case "$1" in
    http://localhost*) echo "http://host.docker.internal${1#http://localhost}" ;;
    https://localhost*) echo "https://host.docker.internal${1#https://localhost}" ;;
    http://127.0.0.1*) echo "http://host.docker.internal${1#http://127.0.0.1}" ;;
    https://127.0.0.1*) echo "https://host.docker.internal${1#https://127.0.0.1}" ;;
    *) echo "$1" ;;
  esac
}
if [ -n "${VLM_BASE_URL:-}" ]; then VLM_BASE_URL="$(host_url "$VLM_BASE_URL")"; export VLM_BASE_URL; fi
if [ -n "${CHAT_BASE_URL:-}" ]; then CHAT_BASE_URL="$(host_url "$CHAT_BASE_URL")"; export CHAT_BASE_URL; fi

exec "$@"

