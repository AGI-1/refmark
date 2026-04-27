# Security Guide

## Token rotation

Service tokens should be rotated every 90 days. Create the replacement token
before revoking the old one, deploy it to dependent services, and verify that
the audit log shows successful authentication with the new credential.

## Audit retention

Security audit logs are retained for 180 days by default. Set
`audit.retention_days` to a larger value when regulatory or customer contracts
require a longer review window.

## Incident escalation

Critical incidents must be acknowledged within 15 minutes. If the primary
owner does not acknowledge the alert, page the secondary owner and open an
incident review record.
