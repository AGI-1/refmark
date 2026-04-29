# Troubleshooting

Start every incident investigation by checking the health endpoint, recent deployments, and the error budget dashboard. This usually separates application regressions from infrastructure or customer traffic problems.

Support bundles collect logs, configuration summaries, and recent job failures. Generate a bundle with `support bundle create --workspace <id>` and attach it to the incident record before escalating to engineering.

Webhook delivery failures usually mean that the customer endpoint timed out, rejected the signature, or returned a permanent HTTP error. Check the retry history, validate the signing secret, and ask the customer to confirm that their endpoint accepts requests from the documented IP ranges.

