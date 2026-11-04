
## Collaboration Workflow: How to Submit a Pull Request

To keep `main` stable and reviewable, all changes must go through a pull request (PR). Direct pushes to `main` are disabled by branch protection.

### Step 1: Create a feature branch
Start from an up-to-date `main` and create a descriptive branch:
```bash
git checkout main
git pull origin main
git checkout -b feature/your-task-name
```

### Step 2: Commit and push your work
Make changes, commit them, and push the branch to the remote repository:
```bash
git add .
git commit -m "Brief description of what you changed"
git push origin feature/your-task-name
```

### Step 3: Open a pull request
1. Go to the repository page on GitHub. A yellow **"Compare & pull request"** button should appear for your recently pushed branch. Click it.
2. Confirm that the **base** branch is `main` and the **compare** branch is your feature branch.
3. Write a clear title and description (e.g., "Add ItemCF implementation and Recall@10 evaluation").
4. Click **Create pull request**.

### Step 4: Request a review
On the PR page, use the **Reviewers** panel on the right to request a review from at least one other group member.

### Step 5: Review and merge
- The reviewer checks the **Files changed** tab, leaves comments if needed, and clicks **Review changes → Approve**.
- Once approved, the PR author (or another group member) clicks **Merge pull request** to merge the changes into `main`.
- Do not merge your own PR without at least one approval.

### Notes
- Keep `main` runnable at all times.
- Pull the latest `main` into your feature branch regularly to reduce merge conflicts.
- Each PR should be small and focused on one task.