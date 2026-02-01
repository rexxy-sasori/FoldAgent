#!/usr/bin/env python
# clone swe-bench-v and swe-gym repo for code search env
#
# pip install datasets gitpython tqdm filelock

import os, shutil, subprocess, tempfile, argparse, json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from filelock import FileLock
from multiprocessing import Pool, cpu_count

# Global dry-run flag
dry_run = False


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download SWE-bench repositories")
parser.add_argument('--base-dir', type=str, default='/root', help='Base directory for storing data and cache')
parser.add_argument('--workers', type=int, default=100, help='Number of workers for parallel processing')
parser.add_argument('--dry-run', action='store_true', help='Calculate size without downloading data')
args = parser.parse_args()

# Set global dry-run flag
dry_run = args.dry_run

# Use configurable base directory for persistent storage
PVC_DIR = Path(args.base_dir)
DATA_DIR = PVC_DIR / "gym_data"  # snapshots per instance
CACHE_DIR = PVC_DIR / "_repo_cache"  # bare/partial repos (shared among workers)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_repo(repo_slug: str) -> Path:
    """
    Return a *partial* bare clone for <github_user/repo>.
    The clone is shared across processes; FileLock prevents races.
    """
    repo_key = repo_slug.replace("/", "__")
    bare_path = CACHE_DIR / f"{repo_key}.git"
    lock_path = bare_path.with_suffix(".lock")

    with FileLock(str(lock_path)):
        if bare_path.exists():
            return bare_path

        url = f"https://github.com/{repo_slug}.git"

        # Try different clone strategies
        clone_strategies = [
            # Strategy 1: Partial clone with blob filter
            lambda: subprocess.run([
                "git", "clone", "--bare", "--filter=blob:none", url, str(bare_path)
            ], check=True, stderr=subprocess.DEVNULL),

            # Strategy 2: Shallow clone with depth
            lambda: subprocess.run([
                "git", "clone", "--bare", "--depth=50", url, str(bare_path)
            ], check=True, stderr=subprocess.DEVNULL),

            # Strategy 3: Full clone (last resort)
            lambda: subprocess.run([
                "git", "clone", "--bare", url, str(bare_path)
            ], check=True, stderr=subprocess.DEVNULL),
        ]

        for i, strategy in enumerate(clone_strategies, 1):
            try:
                strategy()
                break  # Success, exit loop
            except subprocess.CalledProcessError as e:
                if i == len(clone_strategies):
                    # All strategies failed
                    raise RuntimeError(f"All clone strategies failed for {repo_slug}: {e}")
                # Clean up failed attempt
                if bare_path.exists():
                    shutil.rmtree(bare_path, ignore_errors=True)
                continue  # Try next strategy

    return bare_path


def have_commit(repo: Path, sha: str) -> bool:
    """True if <sha> exists in repo."""
    return (
            subprocess.run(
                ["git", "--git-dir", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
    )


def get_commit_size(repo: Path, sha: str) -> int:
    """Calculate the total size of all files in a commit's tree."""
    try:
        # Get all blobs in the commit tree
        blobs = subprocess.run(
            ["git", "--git-dir", str(repo), "ls-tree", "-r", sha],
            capture_output=True,
            text=True,
            stderr=subprocess.DEVNULL
        )
        
        if blobs.returncode != 0:
            return 0
        
        total_size = 0
        for line in blobs.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) < 3:
                continue
            blob_sha = parts[2].split('\t')[0]
            # Get the size of the blob
            size_output = subprocess.run(
                ["git", "--git-dir", str(repo), "cat-file", "-s", blob_sha],
                capture_output=True,
                text=True,
                stderr=subprocess.DEVNULL
            )
            if size_output.returncode == 0:
                try:
                    total_size += int(size_output.stdout.strip())
                except ValueError:
                    pass
        return total_size
    except Exception:
        return 0


def fetch_commit(repo: Path, sha: str):
    """Fetch <sha> into the bare repo *with* blobs for that commit."""
    strategies = [
        # Strategy 1: Try to fetch the specific commit
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "origin", sha
        ], check=True, stderr=subprocess.DEVNULL),

        # Strategy 2: Fetch all branches and tags
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "origin", "+refs/*:refs/*"
        ], check=True, stderr=subprocess.DEVNULL),

        # Strategy 3: Convert to full clone (unshallow)
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "--unshallow"
        ], check=True, stderr=subprocess.DEVNULL),

        # Strategy 4: Complete refetch with full history
        lambda: subprocess.run([
            "git", "--git-dir", str(repo), "fetch", "--all", "--unshallow"
        ], check=True, stderr=subprocess.DEVNULL),
    ]

    for i, strategy in enumerate(strategies, 1):
        try:
            strategy()
            return  # Success, exit early
        except subprocess.CalledProcessError:
            if i == len(strategies):
                # All strategies failed, raise the last error
                raise RuntimeError(f"All fetch strategies failed for commit {sha[:7]}")
            continue  # Try next strategy


def export_commit(repo: Path, sha: str, dest: Path):
    """Export tree for <sha> (fetch first if needed)."""
    if dest.exists():
        return

    # Check if we have the commit, if not try to fetch it
    if not have_commit(repo, sha):
        fetch_commit(repo, sha)
        if not have_commit(repo, sha):
            raise RuntimeError(f"commit {sha[:7]} still missing after fetch")

    tmpdir = Path(tempfile.mkdtemp(dir=dest.parent))
    try:
        tarpath = tmpdir / "repo.tar"
        with open(tarpath, "wb") as f:
            subprocess.run(
                ["git", "--git-dir", str(repo), "archive", sha],
                check=True,
                stdout=f,
            )
        subprocess.run(["tar", "-xf", tarpath, "-C", str(tmpdir)], check=True)
        tarpath.unlink()
        tmpdir.rename(dest)
    finally:
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Per-instance worker
def process_instance(row):
    repo_slug = row["repo"]
    commit_hash = row["base_commit"]
    inst_id = row["instance_id"]
    target_dir = DATA_DIR / inst_id / "testbed"
    
    if dry_run:
        # Calculate size if directory exists
        size = 0
        exists = os.path.exists(target_dir)
        
        if exists:
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    if os.path.isfile(filepath):
                        size += os.path.getsize(filepath)
        else:
            # Try to get size from git repo without creating directories
            try:
                # Get repo path - create bare clone if it doesn't exist
                repo = ensure_repo(repo_slug)
                
                # Check if commit exists locally
                if have_commit(repo, commit_hash):
                    # Calculate size from local repo
                    size = get_commit_size(repo, commit_hash)
                else:
                    # Fetch the commit to calculate its size
                    # This is necessary to get accurate size information
                    fetch_commit(repo, commit_hash)
                    if have_commit(repo, commit_hash):
                        size = get_commit_size(repo, commit_hash)
                    else:
                        # If fetch fails, mark as 0
                        size = 0
            except Exception:
                size = 0
        
        return {
            "status": "dry_run",
            "instance_id": inst_id,
            "repo": repo_slug,
            "commit": commit_hash,
            "size": size,
            "exists": exists
        }
    else:
        # Original processing logic
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(target_dir):
            return {"status": "success", "instance_id": inst_id}

        try:
            repo = ensure_repo(repo_slug)
            export_commit(repo, commit_hash, target_dir)
            return {"status": "success", "instance_id": inst_id}
        except Exception as e:
            return {
                "status": "error",
                "instance_id": inst_id,
                "repo": repo_slug,
                "commit": commit_hash,
                "error": str(e)
            }


# --------------------------------------------------------------------------- #

def main():
    workers = args.workers

    # Load both datasets
    ds1 = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    ds2 = load_dataset("SWE-Gym/SWE-Gym", split="train")
    rows = list(ds1) + list(ds2)  # make it picklable
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Calculating existing data size and providing summary...")
        print(f"Total instances to process: {len(rows)}")
        
        existing_instances = 0
        total_size = 0
        total_download_size = 0
        instances_to_download = 0
        dry_run_results = []
        
        with Pool(processes=workers) as pool:
            for res in tqdm(pool.imap_unordered(process_instance, rows),
                            total=len(rows),
                            desc="Analyzing instances"):
                dry_run_results.append(res)
                if res["exists"]:
                    existing_instances += 1
                    total_size += res["size"]
                else:
                    instances_to_download += 1
                    total_download_size += res["size"]
        
        # Calculate total estimated size
        total_estimated_size = total_size + total_download_size
        
        # Convert size to human-readable format
        def human_readable_size(size):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} PB"
        
        # Summary
        print(f"\n=== DRY RUN SUMMARY ===")
        print(f"Total instances: {len(rows)}")
        print(f"Existing instances: {existing_instances}")
        print(f"Instances to download: {instances_to_download}")
        print(f"Total size of existing data: {human_readable_size(total_size)} ({total_size:,} bytes)")
        print(f"Total estimated size to download: {human_readable_size(total_download_size)} ({total_download_size:,} bytes)")
        print(f"Total estimated size (existing + download): {human_readable_size(total_estimated_size)} ({total_estimated_size:,} bytes)")
        if existing_instances > 0:
            avg_size = total_size / existing_instances
            print(f"Average size per existing instance: {human_readable_size(avg_size)} ({int(avg_size):,} bytes)")
        else:
            print(f"Average size per existing instance: N/A")
        if instances_to_download > 0:
            avg_download_size = total_download_size / instances_to_download
            print(f"Average size per instance to download: {human_readable_size(avg_download_size)} ({int(avg_download_size):,} bytes)")
        else:
            print(f"Average size per instance to download: N/A")
        
        # Breakdown by dataset
        swe_bench_count = len(ds1)
        swe_gym_count = len(ds2)
        print(f"\n=== DATASET BREAKDOWN ===")
        print(f"SWE-bench Verified: {swe_bench_count} instances")
        print(f"SWE-Gym: {swe_gym_count} instances")
        
        return
    else:
        # Original processing logic
        failed_instances = []
        failed_ids = []
        successful_count = 0

        with Pool(processes=workers) as pool:
            for res in tqdm(pool.imap_unordered(process_instance, rows),
                            total=len(rows),
                            desc="Processing instances"):
                if res["status"] == "success":
                    successful_count += 1
                else:
                    failed_instances.append(res)
                    failed_ids.append(res["instance_id"])
                    tqdm.write(f"[ERROR] {res['instance_id']}: {res['error']}")

        # Save failed instances to bad.json in PVC directory
        if failed_instances:
            # Check if bad.json already exists and merge
            existing_failures = []
            bad_json_path = PVC_DIR / "bad.json"
            if bad_json_path.exists():
                try:
                    with open(bad_json_path, "r") as f:
                        existing_failures = json.load(f)
                    print(f"Found {len(existing_failures)} existing failures in bad.json")
                except Exception as e:
                    print(f"Warning: Could not read existing bad.json: {e}")

            # Merge and deduplicate by instance_id
            all_failures = existing_failures + failed_instances
            unique_failures = {}
            for failure in all_failures:
                unique_failures[failure["instance_id"]] = failure

            final_failures = list(unique_failures.values())

            with open(bad_json_path, "w") as f:
                json.dump(final_failures, f, indent=2)
            print(f"\nSaved {len(final_failures)} total failed instances to bad.json")
            print(f"  New failures: {len(failed_instances)}")
            print(f"  Previous failures: {len(existing_failures)}")
        else:
            print("\nNo new failures to save.")

        print(f"\nSummary:")
        print(f"  Successful: {successful_count}")
        print(f"  Failed: {len(failed_instances)}")
        print(f"  Total: {len(rows)}")

        # Print failed instance IDs as a list
        if failed_ids:
            print(f"\nFailed instance IDs:")
            print(failed_ids)


if __name__ == "__main__":
    main()