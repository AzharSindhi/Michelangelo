from pytorch_lightning import Callback
import os
import numpy as np
from git import Repo
from datetime import datetime
from html import escape
import difflib
import tempfile
import wandb

class GitInfoLogger(Callback):
    """
    Callback to save original and reconstructed point clouds as .xyz files after each epoch.
    """
    def __init__(
        self,
        repo_path: str = ".",
    ):
        """
        Args:
            save_dir: Directory to save point cloud files
            max_samples: Maximum number of samples to save per epoch
            every_n_epochs: Save point clouds every N epochs
        """
        super().__init__()
        self.html = [
            '<html><head><meta charset="UTF-8"><title>Git Inline Diff</title>',
            '<style>',
            'body { font-family: monospace; white-space: pre; background: #f8f8f8; padding: 1em; }',
            '.add { background-color: #e6ffed; }',
            '.remove { background-color: #ffeef0; }',
            '.context { color: #aaa; }',
            '.header { font-weight: bold; color: #333; }',
            '</style></head><body>',
            '<h1>Git Inline Diff Report</h1>',
            '<p><i>Includes both tracked and untracked changes.</i></p>'
        ]
        self.repo_path = repo_path
        self.repo = Repo(repo_path)


    def get_git_info(self):

        head_commit = self.repo.head.commit

        # Basic Git info
        commit_info = {
            "git_commit_id": head_commit.hexsha,
            "git_commit_message": head_commit.message.strip(),
            "git_commit_author": f"{head_commit.author.name} <{head_commit.author.email}>",
            "git_commit_date": datetime.fromtimestamp(head_commit.committed_date).isoformat(),
            "git_branch": self.repo.active_branch.name if not self.repo.head.is_detached else "DETACHED_HEAD",
            "git_dirty": self.repo.is_dirty(untracked_files=True),
            # "git_remotes": ", ".join([f"{remote.name} ({list(remote.urls)[0]})"
            #                             for remote in repo.remotes]) or "No remotes",
            # "repo": self.repo  # Return repo separately to use for diff logging
        }
        return commit_info

    def get_untracked_info(self, repo):
        for path in repo.untracked_files:
            abs_path = os.path.abspath(os.path.join(repo.working_tree_dir, path))
            if not abs_path.startswith(repo.working_tree_dir):
                continue
            if os.path.isdir(abs_path):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                working_lines = f.readlines()

            self.html.append(f'<h2 class="header">Untracked: {escape(path)}</h2><pre>')
            for line in working_lines:
                esc = escape(line.rstrip('\n'))
                self.html.append(f'{esc}')
            self.html.append('</pre>')

    def generate_inline_git_diff_html(self, repo):
        # Tracked changes
        for diff_item in repo.index.diff(None):
            path = diff_item.a_path
            abs_path = os.path.abspath(os.path.join(repo.working_tree_dir, path))
            if not abs_path.startswith(repo.working_tree_dir):
                continue
            if os.path.isdir(abs_path):
                continue
            if not os.path.exists(path): # TO-DO: handle deleted files
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                working_lines = f.readlines()
            committed_lines = repo.git.show(f":{path}").splitlines(keepends=True)

            diff = difflib.unified_diff(
                committed_lines, working_lines,
                fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
            )
            self.html.append(f'<h2 class="header">Modified: {escape(path)}</h2><pre>')
            for line in diff:
                esc = escape(line.rstrip('\n'))  # ✅ strip extra newline
                if line.startswith('+') and not line.startswith('+++'):
                    self.html.append(f'<span class="add">{esc}</span>')
                elif line.startswith('-') and not line.startswith('---'):
                    self.html.append(f'<span class="remove">{esc}</span>')
                elif line.startswith(('---', '+++', '@@')):
                    self.html.append(f'<span class="header">{esc}</span>')
                else:
                    self.html.append(f'<span class="context">{esc}</span>')
            self.html.append('</pre>')
        # Untracked files
        self.get_untracked_info(repo)
        self.html.append('</body></html>')

    def log_git_diff(self, logger):

        # Git metadata
        git_info = self.get_git_info()
        if git_info:
            # for key, value in git_info.items():
            #     mlflow_logger.log_tag(key, value)

            # Log patch if working directory is dirty
            if git_info.get("git_dirty"):
                try:
                    self.generate_inline_git_diff_html(self.repo)
                except Exception as e:
                    print(f"❌ Git inline diff failed: {e}")
                
                wandb_html = wandb.Html("\n".join(self.html))
                logger.experiment.log({"git_diff": wandb_html})
        
        print("MLflow run completed with Git metadata.")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = pl_module.logger
        self.log_git_diff(logger)
