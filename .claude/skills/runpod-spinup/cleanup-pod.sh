#!/usr/bin/env bash
# DEPRECATED — pod deletion now goes through the `runpod` MCP server
# (claude mcp add runpod ...) rather than shelling out to runpodctl.
# See SKILL.md. Use ./deregister-ssh-alias.sh <name> after deleting via MCP.
echo "deprecated: delete pods via the 'runpod' MCP server, then run ./deregister-ssh-alias.sh <name>" >&2
exit 1
