# Security policy

Security fixes target the current development branch. Alpha versions and
historical research receipts are not certified deployments. See
[docs/release.md](docs/release.md) for the release gates.

Report vulnerabilities using GitHub's private vulnerability reporting on the
[repository Security page](https://github.com/neverhuman/dope/security/advisories/new).
If private reporting is unavailable, contact a maintainer through their GitHub
profile and ask for a private channel before sending details. Do not put secrets,
private datasets, exploit inputs containing source values, or machine credentials
in a public issue. Include the affected revision, reproduction steps, impact and
a minimized synthetic example when possible.

Maintainers acknowledge reports when available, agree on a disclosure plan with
the reporter, and publish an advisory after a fix can be deployed. There is no
promised response SLA or bug bounty. An empirical privacy evaluation does not
establish formal differential privacy or regulatory de-identification.

Fork pull requests run only on GitHub-hosted runners with read-only repository
permissions and no deployment secrets; `pull_request` workflows on public
repositories run on untrusted fork branches and must not receive write tokens
or secrets. Sealed evaluation labels and vendor weights are excluded from
public smoke tests. Only reviewed, same-repository changes may use private
runners. Keep dependency and workflow checks enabled.
