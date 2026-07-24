#!/usr/bin/env bash
# DEPRECATED — pod creation now goes through the `runpod` MCP server
# (claude mcp add runpod ...) rather than shelling out to runpodctl.
# See SKILL.md. Use ./register-ssh-alias.sh after the MCP create-pod tool
# returns connection details.
echo "deprecated: create pods via the 'runpod' MCP server, then run ./register-ssh-alias.sh <name> <ip> <port>" >&2
exit 1
