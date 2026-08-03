# Git Workflow Guide — animal-welfare-data-pipeline

A simple step-by-step process for making changes and pushing them to the repo.

## 1. Start from an up-to-date `main`

Before starting any new work, make sure you're on `main` and have the latest changes.

```bash
git checkout main
git pull
```

---

## 2. Create a new branch for your work

Never make changes directly on `main`. Create a new branch instead.

```bash
git checkout -b <branch-name>
```

**Naming tip:** use something short and descriptive, ideally with your initials, e.g. `al/fix-constitution-doc` or `al/dev`. This makes it easy to tell whose branch is whose.

---

## 3. Make your changes

Edit files as needed. You can check what's changed at any time with:

```bash
git status
```

---

## 4. Add, commit, and push your changes

```bash
git add .
git commit -m "short description of what you changed"
git push -u origin <branch-name>
```

- `git add .` stages **all** changed files. If you only want to commit specific files, use `git add <file1> <file2>` instead.
- The `-u` flag on your **first** push links your local branch to a matching branch on GitHub, so future pushes/pulls on this branch just need `git push` / `git pull`.

---

## 5. Open a Pull Request (PR)

After pushing, GitHub will print a link in your terminal like:

```
Create a pull request for '<branch-name>' on GitHub by visiting:
     https://github.com/sentfutures/animal-welfare-data-pipeline/pull/new/<branch-name>
```

Click that link (or go to the repo on GitHub — it usually shows a "Compare & pull request" banner) and open the PR. Add a short description of what you changed, then submit it for review.

---

## 6. After the PR is merged: clean up your branches

Once your PR is approved and merged into `main`, delete the branch both on GitHub and locally so things stay tidy.

```bash
# switch back to main first
git checkout main
git pull

# delete the local branch
git branch -d <branch-name>

# delete the remote branch
git push origin --delete <branch-name>
```

You can also delete the remote branch through the GitHub website instead of the command line — after your PR is merged, GitHub usually shows a "Delete branch" button right on the merged PR page. You can also find it under the repo's **branches** tab.

---

## Quick reference

| Command | What it does |
|---|---|
| `git checkout main` | Switch to the `main` branch |
| `git pull` | Download and merge the latest changes from GitHub |
| `git checkout -b <branch-name>` | Create a new branch and switch to it |
| `git add .` | Stage all changed files to be committed |
| `git commit -m "message"` | Save your staged changes with a description |
| `git push -u origin <branch-name>` | Push your branch to GitHub and link it for future pushes/pulls (first push only) |
| *(open PR on GitHub, get it reviewed & merged)* | |
| `git checkout main` | Switch back to `main` |
| `git pull` | Get the newly merged changes |
| `git branch -d <branch-name>` | Delete the branch locally |
| `git push origin --delete <branch-name>` | Delete the branch on GitHub (can also be done via the "Delete branch" button on the merged PR page, or the repo's branches tab) |
