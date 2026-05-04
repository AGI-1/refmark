# Operational Records

| Code | Topic | Operational meaning |
| --- | --- | --- |
| INC-1 | Incident | Customer-visible incident with external communication |
| INC-2 | Incident | Internal incident without customer communication |
| SEC-1 | Security | Privileged access review exception |
| SEC-2 | Security | Service token rotation exception |

## Duplicate Support Note

Service tokens are used by automation jobs. Rotate tokens every ninety days and
store them in the managed secret store rather than in source code.

## Escalation Record

An urgent incident should include impact, affected service, first detected time,
and whether customer data may be involved before escalation.
