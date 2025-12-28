# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

We take the security of dRoute seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**darri.eythorsson@ucalgary.ca**

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### What to Expect

- You should receive an acknowledgment of your report within 48 hours
- We will investigate the issue and determine its severity
- We will work on a fix and coordinate disclosure timing with you
- Once the vulnerability is fixed, we will publish a security advisory
- We will credit you for the discovery (unless you prefer to remain anonymous)

### Safe Harbor

We support responsible disclosure. If you make a good faith effort to comply with this policy:

- We will not pursue legal action against you
- We will work with you to understand and resolve the issue quickly
- We will credit you for the discovery (with your permission)

## Security Best Practices

When using dRoute:

1. **Input Validation**: Always validate input data (network topology, runoff values, etc.)
2. **Dependencies**: Keep dependencies up to date, especially NumPy and PyTorch
3. **File I/O**: Be cautious when loading topology files from untrusted sources
4. **Numerical Stability**: Be aware that extreme parameter values may cause numerical issues

## Disclosure Policy

- Security vulnerabilities will be disclosed publicly after a fix is available
- We aim to release security patches within 90 days of receiving a report
- Critical vulnerabilities will be prioritized and patched as quickly as possible

## Comments

If you have suggestions on how this process could be improved, please submit a pull request or email us.

## Acknowledgments

We would like to thank the following individuals for responsibly disclosing security vulnerabilities:

- (None yet)
