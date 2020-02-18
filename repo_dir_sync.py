# -*- coding: utf-8 -*-
import os
import shutil
from subprocess import Popen, PIPE
import re
import sys
from typing import List


def call(cmd):
    p = Popen(cmd, shell=False, stdin=PIPE, stdout=PIPE)
    return p.stdout.read().decode("utf-8")


def execute(cmd):
    ret = os.system(cmd)
    if ret != 0:
        raise Exception("Command failed: %s" % cmd)


def gitLog(path, arg):
    oldPath = os.getcwd()
    os.chdir(path)
    lg = call("git log --no-merges " + arg)
    os.chdir(oldPath)
    return lg


def gitCommit(msg):
    with open("commitmsg.txt", "w") as f:
        f.write(msg)
    os.system("git commit --file=commitmsg.txt")
    os.unlink("commitmsg.txt")


LIB_DIRECTORY = "basic_models"
LIB_NAME = "basic_models"


class RemoteRepo:
    SYNC_COMMIT_ID_FILE_LIB_REPO = ".syncCommitId"
    SYNC_COMMIT_ID_FILE_THIS_REPO = ".syncCommitId.this"
    
    def __init__(self, name, branch, pathToBasicModels):
        self.pathToLibInThisRepo = os.path.abspath(pathToBasicModels)
        if not os.path.exists(self.pathToLibInThisRepo):
            raise ValueError(f"Repository directory '{self.pathToLibInThisRepo}' does not exist")
        self.name = name
        self.branch = branch
    
    def lastSyncIdThisRepo(self):
        with open(os.path.join(self.pathToLibInThisRepo, self.SYNC_COMMIT_ID_FILE_THIS_REPO), "r") as f:
            commitId = f.read().strip()
        return commitId
    
    def gitLogThisRepoSinceLastSync(self):
        lg = gitLog(self.pathToLibInThisRepo, '--name-only HEAD "^%s" .' % self.lastSyncIdThisRepo())
        lg = re.sub(r'commit [0-9a-z]{8,40}\n.*\n.*\n\s*\n.*\n\s*\n.*\.syncCommitId\.this', r"", lg, flags=re.MULTILINE)  # remove commits with sync commit id update
        indent = "  "
        lg = indent + lg.replace("\n", "\n" + indent)
        return lg

    def gitLogLibRepoSinceLastSync(self, libRepo: "LibRepo"):
        syncIdFile = os.path.join(self.pathToLibInThisRepo, self.SYNC_COMMIT_ID_FILE_LIB_REPO)
        if not os.path.exists(syncIdFile):
            return ""
        with open(syncIdFile, "r") as f:
            syncId = f.read().strip()
        lg = gitLog(libRepo.libPath, 'HEAD "^%s" .'  % syncId)
        lg = re.sub(r"Sync (\w+)\n\s*\n", r"Sync\n\n", lg, flags=re.MULTILINE)
        indent = "  "
        lg = indent + lg.replace("\n", "\n" + indent)
        return "\n\n" + lg

    def pull(self, libRepo: "LibRepo"):
        """
        Pulls in changes from this remote repository into the lib repo
        """
        os.chdir(libRepo.rootPath)

        # switch to branch in lib repo and remove library tree
        execute("git checkout %s" % self.branch)
        shutil.rmtree(LIB_DIRECTORY)

        # get log with relevant commits in this repo
        lg = self.gitLogThisRepoSinceLastSync()

        # copy tree from this repo to lib repo
        shutil.copytree(self.pathToLibInThisRepo, LIB_DIRECTORY)
        for fn in (self.SYNC_COMMIT_ID_FILE_LIB_REPO, self.SYNC_COMMIT_ID_FILE_THIS_REPO):
            p = os.path.join(LIB_DIRECTORY, fn)
            if os.path.exists(p):
                os.unlink(p)

        # make commit in lib repo
        os.system("git add %s" % LIB_DIRECTORY)
        gitCommit(f"Sync {self.name}\n\n" + lg)
        newSyncCommitIdLibRepo = call("git rev-parse HEAD").strip()

        # update commit ids in this repo
        os.chdir(self.pathToLibInThisRepo)
        newSyncCommitIdThisRepo = call("git rev-parse HEAD").strip()
        with open(self.SYNC_COMMIT_ID_FILE_LIB_REPO, "w") as f:
            f.write(newSyncCommitIdLibRepo)
        with open(self.SYNC_COMMIT_ID_FILE_THIS_REPO, "w") as f:
            f.write(newSyncCommitIdThisRepo)
        execute('git add %s %s' % (self.SYNC_COMMIT_ID_FILE_LIB_REPO, self.SYNC_COMMIT_ID_FILE_THIS_REPO))
        execute('git commit -m "Updated sync commit identifiers (pull)"')

        print(f"\n\nIf everything was successful, you should now try to merge '{self.branch}' into master:\ngit push\ngit checkout master; git merge {self.branch}\ngit push")
        
    def push(self, libRepo: "LibRepo"):
        """
        Pushes changes from the lib repo to this remote repo
        """

        # get change log since last sync
        libLogSinceLastSync = self.gitLogLibRepoSinceLastSync(libRepo)

        os.chdir(libRepo.rootPath)

        # check if this repo's branch in the source repo was merged into master
        unmergedBranches = call("git branch --no-merged master")
        if self.branch in unmergedBranches:
            raise Exception(f"Branch {self.branch} has not been merged into master")

        # switch to the source repo branch and merge master into it (to make sure it's up to date)
        execute(f"git checkout {self.branch}")
        execute("git merge master")

        # remove the target repo tree and update it with the tree from the source repo
        shutil.rmtree(self.pathToLibInThisRepo)
        shutil.copytree(libRepo.libPath, self.pathToLibInThisRepo)

        # get the commit id of the source repo we just copied
        commitId = call("git rev-parse HEAD").strip()

        os.chdir(self.pathToLibInThisRepo)

        # commit new version in this repo
        execute("git add .")
        with open(self.SYNC_COMMIT_ID_FILE_LIB_REPO, "w") as f:
            f.write(commitId)
        execute("git add %s" % self.SYNC_COMMIT_ID_FILE_LIB_REPO)
        gitCommit(f"{LIB_NAME} {commitId}" + libLogSinceLastSync)
        commitId = call("git rev-parse HEAD").strip()

        # update information on the commit id we just added
        with open(self.SYNC_COMMIT_ID_FILE_THIS_REPO, "w") as f:
            f.write(commitId)
        execute("git add %s" % self.SYNC_COMMIT_ID_FILE_THIS_REPO)
        execute('git commit -m "Updated sync commit identifier (push)"')

        os.chdir(libRepo.rootPath)
        
        print(f"\n\nIf everything was successful, you should now update the remote branch:\ngit push")


class LibRepo:
    def __init__(self):
        self.rootPath = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        self.libPath = os.path.join(self.rootPath, LIB_DIRECTORY)
        self.remoteRepos: List[RemoteRepo] = []

    def add(self, repo: RemoteRepo):
        self.remoteRepos.append(repo)

    def runMain(self):
        repos = self.remoteRepos
        args = sys.argv[1:]
        if len(args) != 2:
            print(f"usage: sync.py <{'|'.join([repo.name for repo in repos])}> <push|pull>")
        else:
            repo = [r for r in repos if r.name == args[0]]
            if len(repo) != 1:
                raise ValueError(f"Unknown repo '{args[0]}'")
            repo = repo[0]

            if args[1] == "push":
                repo.push(self)
            elif args[1] == "pull":
                repo.pull(self)
            else:
                raise ValueError(f"Unknown command '{args[1]}'")