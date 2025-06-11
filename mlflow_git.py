import os
import mlflow
from git import Repo, InvalidGitRepositoryError, GitCommandError
from datetime import datetime
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import HtmlFormatter
import difflib
from html import escape
import tempfile

def get_git_info(repo_path="."):

    repo = Repo(repo_path)
    head_commit = repo.head.commit

    # Basic Git info
    commit_info = {
        "git_commit_id": head_commit.hexsha,
        "git_commit_message": head_commit.message.strip(),
        "git_commit_author": f"{head_commit.author.name} <{head_commit.author.email}>",
        "git_commit_date": datetime.fromtimestamp(head_commit.committed_date).isoformat(),
        "git_branch": repo.active_branch.name if not repo.head.is_detached else "DETACHED_HEAD",
        "git_dirty": repo.is_dirty(untracked_files=True),
        "git_remotes": ", ".join([f"{remote.name} ({list(remote.urls)[0]})"
                                    for remote in repo.remotes]) or "No remotes",
        "repo": repo  # Return repo separately to use for diff logging
    }
    return commit_info


def generate_inline_git_diff_html(mlflow_logger,repo, filename="git_diff_inline.html"):
    try:
        html = [
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

        # Tracked changes
        for diff_item in repo.index.diff(None):
            path = diff_item.a_path
            abs_path = os.path.abspath(os.path.join(repo.working_tree_dir, path))
            if not abs_path.startswith(repo.working_tree_dir):
                continue
            if os.path.isdir(abs_path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    working_lines = f.readlines()
                committed_lines = repo.git.show(f":{path}").splitlines(keepends=True)

                diff = difflib.unified_diff(
                    committed_lines, working_lines,
                    fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
                )
                html.append(f'<h2 class="header">Modified: {escape(path)}</h2><pre>')
                for line in diff:
                    esc = escape(line.rstrip('\n'))  # ✅ strip extra newline
                    if line.startswith('+') and not line.startswith('+++'):
                        html.append(f'<span class="add">{esc}</span>')
                    elif line.startswith('-') and not line.startswith('---'):
                        html.append(f'<span class="remove">{esc}</span>')
                    elif line.startswith(('---', '+++', '@@')):
                        html.append(f'<span class="header">{esc}</span>')
                    else:
                        html.append(f'<span class="context">{esc}</span>')
                html.append('</pre>')
            except Exception as e:
                html.append(f'<p style="color:red;">[Error reading tracked file {path}]: {e}</p>')

        # Untracked files
        for path in repo.untracked_files:
            abs_path = os.path.abspath(os.path.join(repo.working_tree_dir, path))
            if not abs_path.startswith(repo.working_tree_dir):
                continue
            if os.path.isdir(abs_path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    working_lines = f.readlines()

                html.append(f'<h2 class="header">Untracked: {escape(path)}</h2><pre>')
                for line in working_lines:
                    esc = escape(line.rstrip('\n'))
                    html.append(f'{esc}')
                html.append('</pre>')

            except Exception as e:
                html.append(f'<p style="color:red;">[Error reading untracked file {path}]: {e}</p>')

        html.append('</body></html>')
        return html

    except Exception as e:
        print(f"❌ Failed to generate inline Git diff: {e}")

def log_git_details(mlflow_logger):

        # Git metadata
        git_info = get_git_info()
        if git_info:
            repo = git_info.pop("repo")
            for key, value in git_info.items():
                mlflow.set_tag(key, value)

            # Log patch if working directory is dirty
            if git_info.get("git_dirty"):
                html = generate_inline_git_diff_html(mlflow_logger,repo)
                with tempfile.NamedTemporaryFile(delete=True, prefix="git_diff", suffix=".html") as tmpfile:
                    tmpfile.write("\n".join(html).encode("utf-8"))  
                    mlflow.log_artifact(tmpfile.name)
                    print(f"✅ Git inline diff saved and logged as: {tmpfile.name}")
        
        print("MLflow run completed with Git metadata.")

# if __name__ == "__main__":
#     with mlflow.start_run() as mlflow_logger:
#         log_git_details(mlflow_logger)
