import logging
from dataclasses import dataclass
import subprocess
from typing import Optional

from .string import ToStringMixin

log = logging.getLogger(__name__)


@dataclass
class GitStatus(ToStringMixin):
    commit: str
    has_unstaged_changes: bool
    has_staged_uncommitted_changes: bool
    has_untracked_files: bool

    @property
    def is_clean(self) -> bool:
        return not (self.has_unstaged_changes or
                    self.has_staged_uncommitted_changes or
                    self.has_untracked_files)


def git_status() -> Optional[GitStatus]:
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        unstaged = bool(subprocess.check_output(['git', 'diff', '--name-only']).decode('ascii').strip())
        staged = bool(subprocess.check_output(['git', 'diff', '--staged', '--name-only']).decode('ascii').strip())
        untracked = bool(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard']).decode('ascii').strip())
        return GitStatus(
            commit=commit_hash,
            has_unstaged_changes=unstaged,
            has_staged_uncommitted_changes=staged,
            has_untracked_files=untracked
        )
    except Exception as e:
        log.error("Error determining Git status", exc_info=e)
        return None
