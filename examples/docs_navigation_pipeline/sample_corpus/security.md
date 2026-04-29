# Security

API tokens identify callers and should be scoped to the smallest useful permission set. A token used by a deployment pipeline should not also be used by a billing integration or a support tool.

Rotate API tokens every ninety days. Create the replacement token before revoking the old token, deploy it to dependent services, verify successful authentication in the audit log, and only then revoke the old token.

Audit logs are retained for one hundred eighty days by default. Increase `audit.retention_days` when a customer contract, compliance program, or internal policy requires a longer review window.

