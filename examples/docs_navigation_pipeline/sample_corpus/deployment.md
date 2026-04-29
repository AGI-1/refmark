# Deployment

Deployments use a rolling strategy. Start by draining background workers, apply database migrations that are backwards compatible, deploy the new application image, and then restart workers after the health checks pass.

Every release must have a rollback plan. Keep the previous container image available for at least twenty-four hours, record any migration that cannot be reversed automatically, and verify that the old image can read data written by the new image.

Production releases should be scheduled between 09:00 and 15:00 UTC. Emergency changes outside that window require approval from the incident commander and a written follow-up note.

