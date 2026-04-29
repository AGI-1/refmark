# Authentication

The application supports password login, single sign-on, and service accounts. Password login is intended for small teams and local development. Production workspaces should configure single sign-on so access policies can be managed from the company identity provider.

Single sign-on uses SAML 2.0. Set `auth.saml.enabled` to `true`, upload the identity provider metadata, and map identity provider groups to workspace roles with `auth.saml.group_map`. Users are created on first successful login when just-in-time provisioning is enabled.

Service accounts are non-human identities for automation. Create one service account per integration, assign the minimum required role, and rotate its token on the same schedule as other API credentials.

