# Container tag facts (task E)

**Established 2026-08-26 by direct inspection. Nothing was built, tagged, pushed or published.**

The question: `containers/magicc_0.3.0.sif` exists on disk, `REPO_RELEASE_RECORD.md`
names `containers/magicc_0.3.1.sif`, and `supplementary_revised.md` refers to a Docker
image `magicc:0.3.0`. What actually exists?

---

## 1. Apptainer / Singularity SIF — only 0.3.0 exists

| Fact | Value | Evidence |
|---|---|---|
| Files matching `containers/*.sif` | **exactly one: `magicc_0.3.0.sif`** | `ls -la containers/*.sif` |
| Size | 384,008,192 B (366 MiB, 0.384 GB) | `ls -la` |
| SHA256 | `287e83d43549594fe580dd8b3cb2ba2d210a706b01f7b41673c8d26948c7aa04` | `sha256sum` |
| mtime | 2026-07-26 16:56 | `ls -la` |
| Embedded build date label | `Sunday_26_July_2026_16:56:23_CDT` | SIF label block |
| Embedded `deffile` | `bootstrap: docker-daemon` / `from: magicc:0.3.0` | SIF label block |
| `org.opencontainers.image.version` | **`0.3.0`** | SIF label block |
| `org.magicc.model.sha256` | `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096` | SIF label block |
| `org.magicc.model.version` | `V5` | SIF label block |
| Built with | apptainer 1.4.5, arch amd64 | SIF label block |

**`containers/magicc_0.3.1.sif` does not exist.** It was never built. The SIF on disk
carries the tag **0.3.0** in its own embedded metadata, so it cannot be renamed into a
0.3.1 artefact without rebuilding — the label block would still say 0.3.0.

### The `REPO_RELEASE_RECORD.md` row is wrong

Its "deliberately NOT published" table contains:

> `containers/magicc_0.3.1.sif` | 367 MB | rebuildable from the tracked definition

That row implies a 0.3.1 SIF exists locally and was withheld. **No such file exists.**
The quoted 367 MB is the size of the **0.3.0** SIF (366 MiB), carried forward under a
0.3.1 name. The row should read `containers/magicc_0.3.0.sif | 366 MiB | the only SIF
built; a 0.3.1 SIF was never built`, or the SIF should actually be rebuilt at 0.3.1.

### The tracked definition files were already renamed to 0.3.1

`containers/magicc.def` line 39 (`From: magicc:0.3.1`), line 43
(`org.opencontainers.image.version 0.3.1`), and `containers/build_containers.sh`
lines 17-18 (`TAG="magicc:0.3.1"`, `SIF="${REPO}/containers/magicc_0.3.1.sif"`) all name
0.3.1. So the **recipe** says 0.3.1 while the **only built artefact** says 0.3.0. The
definition would produce the 0.3.1 SIF correctly if run; it simply has not been run.

Note the standing WS7 limitation is unchanged: Apptainer **cannot execute** on this host
(`kernel.apparmor_restrict_unprivileged_userns=1`, no setuid starter, no passwordless
sudo — re-confirmed today: `unshare -rn` fails with `Operation not permitted`).

---

## 2. Docker images — BOTH 0.3.0 and 0.3.1 exist locally

| Tag | Image ID | Created | `image.version` label | `org.magicc.model.sha256` | RepoDigests |
|---|---|---|---|---|---|
| `magicc:0.3.0` | `sha256:d13e68ead3b6…` | 2026-07-26T16:45:02-05:00 | `0.3.0` | `b84346…b3096` | **`[]`** |
| `magicc:0.3.1` | `sha256:38804a236e32…` | 2026-08-25T11:56:57-05:00 | `0.3.1` | `b84346…b3096` | **`[]`** |

Both carry the correct V5 model hash. The `magicc:0.3.1` image ID matches the one
`REPO_RELEASE_RECORD.md` §4 records (`sha256:38804a236e32…`), so the 0.3.1 Docker image
**was** built on release day, contrary to what the SIF situation suggests.

**Empty `RepoDigests` on both images means neither has ever been pushed to or pulled from
any registry.** A registry digest is recorded the moment an image is pushed or pulled;
its absence is direct evidence of purely local existence.

---

## 3. Is any image published anywhere? — No

| Registry | Probe | Result |
|---|---|---|
| Docker Hub | `GET hub.docker.com/v2/repositories/renmaotian/magicc/` | **404** (the namespace itself returns 200, so the account exists and the repository does not) |
| GHCR | `GET ghcr.io/v2/renmaotian/magicc/tags/list` with an anonymous pull token | **403** — not publicly readable |
| GitHub Packages | `GET /users/renmaotian/packages?package_type=container` (authenticated) | **`[]`** — the user owns no container packages |
| quay.io | `GET quay.io/api/v1/repository/renmaotian/magicc` | **401** |
| GitHub release `v0.3.1` | release API | **0 uploaded assets** (only GitHub's auto-generated source tarballs) |

**No MAGICC container image is published anywhere.** Both Docker images and the single
SIF exist only on this host.

---

## 4. Plain statement for the manuscript and supplementary

1. **No container is published.** Any sentence implying a reader can pull a MAGICC image
   is false. What ships is a **Dockerfile and an Apptainer definition**, from which a
   reader builds the image themselves. `manuscript_revised.md` line 308 already words it
   correctly ("a Docker image … an Apptainer/Singularity definition built from it … are
   provided"), but "provided" should be unambiguously readable as *the recipe is
   provided*, not *the image is hosted*.
2. **The Docker image tag that shipping definitions produce is `magicc:0.3.1`**, and that
   image has been built locally and verified.
3. **The only Apptainer artefact ever built is `magicc_0.3.0.sif`**, whose own embedded
   labels say `0.3.0`. Either rebuild it at 0.3.1, or state 0.3.0 with explicit
   historical framing.
4. **Table S8d's cross-environment determinism probe genuinely ran on `magicc:0.3.0`**,
   which is why the audit's decision to keep that string with historical framing is
   correct. It should not be silently rewritten to 0.3.1 — that would misdescribe which
   image was measured. (The 0.3.0 image lacks `--input-list`, so a 0.3.1 rerun would be a
   different, stronger probe — but it has not been run.)
5. `REPO_RELEASE_RECORD.md`'s `containers/magicc_0.3.1.sif | 367 MB` row must be
   corrected: that file does not exist.

**Nothing was built, retagged or published in producing this record.**
