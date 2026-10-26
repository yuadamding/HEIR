# Pre-cleanup source archive

[`history.tar.xz`](history.tar.xz) preserves the complete source snapshot before repository
compaction, including tracked files, uncommitted changes, new files, the original README, and
historical scientific reports and receipts. Active implementation files are also included as they
existed at that point. The snapshot excludes `.git`, virtual environments, generated caches, raw
datasets, and model checkpoints.

| Identity | Value |
| --- | --- |
| Source commit | `ca7bb184406eb3145b4fb7559f9b3a29c6906d9b` |
| Source files | 273 |
| Source bytes | 6,447,445 |
| Compressed bytes | 1,030,688 |
| SHA-256 | `45e8d5c4a64061f0165d2cf2f2806fe0910623afc786bcc96a4b18c580a7d193` |

The internal `ARCHIVE_MANIFEST.json` records the source commit, working-tree status, original
relative paths, sizes, and per-file hashes. The snapshot contains uncommitted work, so the commit
alone does not reproduce it. All 273 archived source files were checked against their original
contents before removal from the active tree.

From the repository root, verify the archive hash:

```bash
sha256sum archive/history.tar.xz
```

To inspect or recover it, choose a separate empty directory:

```bash
mkdir /tmp/heir-history
tar -xJf archive/history.tar.xz -C /tmp/heir-history
```

Original paths are retained inside that directory. The historical README links to its reports,
protocols, scripts, and source files there. Extracting over the active worktree would overwrite
current files; historical dependencies and execution instructions belong to the archived snapshot.

Archival does not alter failed findings, restore exposed cohorts to prospective status, reopen
terminal studies, or authorize historical execution. HEST, NatCommun, and GSE315411 receipts and
their original claim boundaries remain preserved in the snapshot.
