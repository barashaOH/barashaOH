# Incident Response Mini Playbook

1. **Identify**: validate alert fidelity, affected host(s), user account, and timeline.
2. **Contain**: isolate endpoint or segment; block suspicious IP/domain at network controls.
3. **Eradicate**: remove malicious artifacts; reset credentials and revoke tokens.
4. **Recover**: patch exploited systems; restore trusted backups if needed.
5. **Lessons learned**: write post-incident review with root cause and control gaps.

## Ransomware immediate actions
- Disconnect impacted machine from network.
- Preserve volatile evidence when possible.
- Disable lateral movement paths (RDP, SMB shares, privileged accounts).
- Notify legal/compliance and incident commander.
