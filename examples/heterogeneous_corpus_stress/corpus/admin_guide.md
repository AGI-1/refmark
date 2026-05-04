# Admin Guide

## Password Reset

Administrators can reset a user's password from the account security panel. The
reset action creates a temporary sign-in link that expires after fifteen
minutes. Support staff should never ask for the old password.

## Audit Export

The audit export includes actor, timestamp, IP address, action name, and target
resource. Security teams use this export when investigating suspicious account
activity.

## Service Tokens

Service tokens are used by automation jobs. Rotate tokens every ninety days and
store them in the managed secret store rather than in source code.
