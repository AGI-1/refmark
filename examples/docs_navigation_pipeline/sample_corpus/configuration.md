# Configuration

Configuration can be supplied through environment variables, a YAML file, or command line flags. The loader applies defaults first, then the YAML file, then environment variables, and finally explicit flags. Later sources override earlier sources.

Rate limits are controlled by `api.rate_limit_per_minute` and `api.burst_limit`. The minute limit protects shared capacity, while the burst limit absorbs short spikes from user interfaces and scheduled jobs. Set both values per workspace when customers have different traffic profiles.

Secrets should come from the deployment environment or a secret manager. Do not store access tokens, private keys, or webhook signing secrets in the YAML file committed with the application.

