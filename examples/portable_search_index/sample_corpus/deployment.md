# Deployment Notes

## Release windows

Production releases are normally scheduled between 09:00 and 15:00 UTC. Avoid
database migrations during customer peak traffic unless an emergency change has
been approved.

## Rollback

Every deployment must include a rollback plan. Keep the previous container
image available for at least 24 hours and record any schema changes that cannot
be reversed automatically.
