# Sandbox image for tools/py_sandbox.py (AGENTS.md §9.3).
#
# Build once, on a machine that still has a network, BEFORE arming the firewall:
#     docker build -t sandbox-py:local sandbox/
#
# After that the image is on disk and the container runs with --network none,
# so nothing here is fetched at demo time (§2.1: weights and images are
# pre-staged, runtime never downloads).
#
# Deliberately stdlib-only. Adding numpy/pandas would mean a bigger image, a
# longer offline build, and a supply chain to vouch for — and every §14.5 task
# (thickness maths, corrosion rate, remaining life) is arithmetic that the
# standard library already does.

FROM python:3.13-slim

# The container also runs --read-only with a tmpfs at /work, so this user needs
# no writable home. -m is omitted on purpose.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
USER 10001

# Unbuffered so stdout survives a kill at the 60 s timeout.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /work
CMD ["python", "-c", "print('sandbox ready')"]
