# Webhooks

Webhooks notify external systems about events such as invoice creation, user provisioning, and policy changes. Each endpoint receives only the event types selected in the workspace settings.

Audit event streaming sends security-sensitive events to a customer SIEM. Configure a webhook endpoint with the `audit.event` topic, store the signing secret in the secret manager, and verify the first delivery before enabling production alerts.

Failed webhook deliveries are retried with exponential backoff. Permanent errors disable the endpoint after repeated failures so one broken integration cannot block the delivery queue for other customers.

