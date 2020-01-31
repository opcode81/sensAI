# -*- coding: utf-8 -*-
import os
import shutil
from subprocess import Popen, PIPE
import re
import sys


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
    lg = call("git log --no-merges --name-only " + arg)
    os.chdir(oldPath)
    return lg


def gitCommit(msg):
    with open("commitmsg.txt", "w") as f:
        f.write(msg)
    os.system("git commit --file=commitmsg.txt")
    os.unlink("commitmsg.txt")


LIB_DIRECTORY = "basic_models"
LIB_NAME = LIB_DIRECTORY

libRepoRootPath = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
libRepoLibPath = os.path.join(libRepoRootPath, LIB_DIRECTORY)


class Repo:
    SYNC_FILE_BASIC_MODELS_REPO = ".syncCommitId"
    SYNC_FILE_THIS_REPO = ".syncCommitId.this"
    
    def __init__(self, name, branch, pathToBasicModels):
        self.pathToLibInThisRepo = os.path.abspath(pathToBasicModels)
        if not os.path.exists(self.pathToLibInThisRepo):
            raise ValueError(f"Repository directory '{self.pathToLibInThisRepo}' does not exist")
        self.name = name
        self.branch = branch
    
    def lastSyncIdThisRepo(self):
        with open(os.path.join(self.pathToLibInThisRepo, self.SYNC_FILE_THIS_REPO), "r") as f:
            commitId = f.read().strip()
        return commitId
    
    def gitLogThisRepoSinceLastSync(self):
        lg = gitLog(self.pathToLibInThisRepo, 'HEAD "^%s" .' % self.lastSyncIdThisRepo())
        lg = re.sub(r'commit [0-9a-z]{8,40}\n.*\n.*\n\s*\n.*\n\s*\n.*\.syncCommitId\.this', r"", lg, flags=re.MULTILINE)  # remove commits with sync commit id update
        indent = "  "
        lg = indent + lg.replace("\n", "\n" + indent)
        return lg

    def gitLogLibRepoSinceLastSync(self):
        syncIdFile = os.path.join(self.pathToLibInThisRepo, self.SYNC_FILE_BASIC_MODELS_REPO)
        if not os.path.exists(syncIdFile):
            return ""
        with open(syncIdFile, "r") as f:
            syncId = f.read().strip()
        lg = gitLog(libRepoLibPath, 'HEAD "^%s" .'  % syncId)
        lg = re.sub(r"Sync (\w+)\n\n", r"Sync\n\n", lg, flags=re.MULTILINE)
        indent = "  "
        lg = indent + lg.replace("\n", "\n" + indent)
        return "\n\n" + lg

    def pull(self):
        os.chdir(libRepoRootPath)
        execute("git checkout %s" % self.branch)
        shutil.rmtree(LIB_DIRECTORY)
        lg = self.gitLogThisRepoSinceLastSync()
        shutil.copytree(self.pathToLibInThisRepo, LIB_DIRECTORY)
        for fn in (self.SYNC_FILE_BASIC_MODELS_REPO, self.SYNC_FILE_THIS_REPO):
            p = os.path.join(LIB_DIRECTORY, fn)
            if os.path.exists(p):
                os.unlink(p)
        os.system("git add %s" % LIB_DIRECTORY)
        gitCommit(f"Sync {self.name}\n\n" + lg)
        print(f"\n\nIf everything was successful, you should now try to merge '{self.branch}' into master:\ngit push\ngit checkout master; git merge {self.branch}\ngit push")
        
    def push(self):
        os.chdir(libRepoRootPath)

        # check if this repo's branch in the source repo was merged into master
        unmergedBranches = call("git branch --no-merged master")
        if self.branch in unmergedBranches:
            raise Exception(f"Branch {self.branch} has not been merged into master")

        # switch to the source repo branch and merge master into it (to make sure it's up to date)
        execute(f"git checkout {self.branch}")
        execute("git merge master")

        # remove the target repo tree and update it with the tree from the source repo
        shutil.rmtree(self.pathToLibInThisRepo)
        shutil.copytree(libRepoLibPath, self.pathToLibInThisRepo)

        # get the commit id of the source repo we just copied
        commitId = call("git rev-parse HEAD").strip()
        
        os.chdir(self.pathToLibInThisRepo)

        # commit new version in this repo
        execute("git add .")
        with open(self.SYNC_FILE_BASIC_MODELS_REPO, "w") as f:
            f.write(commitId)
        execute("git add %s" % self.SYNC_FILE_BASIC_MODELS_REPO)
        gitCommit(f"{LIB_NAME} {commitId}" + self.gitLogLibRepoSinceLastSync())
        commitId = call("git rev-parse HEAD").strip()

        # update information on the commit id we just added
        with open(self.SYNC_FILE_THIS_REPO, "w") as f:
            f.write(commitId)
        execute("git add %s" % self.SYNC_FILE_THIS_REPO)
        execute('git commit -m "Updated sync commit identifier"')

        os.chdir(libRepoRootPath)
        
        print(f"\n\nIf everything was successful, you should now update the remote branch:\ngit push")
    

if __name__ == '__main__':
    odm = Repo("odm", "odm", os.path.join("..", "odm", "odmalg", "basic_models"))
    faz = Repo("faz", "faz", os.path.join("..", "faz", "preprocessing", "basic_models"))
    dcs = Repo("dcs", "dcs", os.path.join("..", "dcs", "dcs-models", "dcs", "basic_models"))
    repos = [odm, faz, dcs]
    
    args = sys.argv[1:]
    if len(args) != 2:
        print(f"usage: sync.py <{'|'.join([repo.name for repo in repos])}> <push|pull>")
        sys.exit(0)
        
    repo = [r for r in repos if r.name == args[0]]
    if len(repo) != 1:
        raise ValueError(f"Unknown repo '{args[0]}'")
    repo = repo[0]
    
    if args[1] == "push":
        repo.push()
    elif args[1] == "pull":
        repo.pull()
    else:
        raise ValueError(f"Unknown command '{args[1]}'")