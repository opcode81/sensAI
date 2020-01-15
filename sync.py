# -*- coding: utf-8 -*-
import os
import shutil
from subprocess import Popen, PIPE
import re
import sys


def call(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE)
    return p.stdout.read().decode("utf-8")


def execute(cmd):
    ret = os.system(cmd)
    if ret != 0:
        raise Exception("Command failed: %s" % cmd)


def gitlog(path, arg):
    oldPath = os.getcwd()
    os.chdir(path)
    lg = call("git log --no-merges --name-only " + arg)
    os.chdir(oldPath)
    return lg


homePath = os.path.abspath(".")


class Repo:
    def __init__(self, name, branch, pathToBasicModels):
        self.pathToBasicModels = os.path.abspath(pathToBasicModels)
        if not os.path.exists(self.pathToBasicModels):
            raise ValueError(f"Repository directory '{self.pathToBasicModels}' does not exist")
        self.name = name
        self.branch = branch
    
    def gitlog(self):
        return gitlog(self.pathToBasicModels, ".")
    
    def gitlogSinceSync(self):
        lg = self.gitlog()
        lg = re.sub(r'basic_models ([0-9a-z]{8,40}).*', r"basic_models \1", lg, flags=re.MULTILINE | re.DOTALL)
        indent = "  "
        lg = indent + lg.replace("\n", "\n" + indent)
        return lg
    
    def pull(self):
        os.chdir(homePath)
        execute("git checkout %s" % self.branch)
        shutil.rmtree("basic_models")
        #os.mkdir("basic_models")
        shutil.copytree(self.pathToBasicModels, "basic_models")
        lg = self.gitlogSinceSync()
        os.system("git add basic_models")
        with open("commitmsg.txt", "w") as f:
            f.write(f"Sync {self.name}\n\n")
            f.write(lg)
        os.system("git commit --file=commitmsg.txt")
        print(f"\n\nIf everything was successful, you should now try to merge '{self.branch}' into master:\ngit push\ngit checkout master; git merge {self.branch}\ngit push")
        
    def push(self):
        os.chdir(homePath)
        unmergedBranches = call("git branch --no-merged master")
        if self.branch in unmergedBranches:
            raise Exception(f"Branch {self.branch} has not been merged into master")
        execute(f"git checkout {self.branch}")
        execute("git merge master")
        shutil.rmtree(self.pathToBasicModels)
        shutil.copytree(os.path.join(homePath, "basic_models"), self.pathToBasicModels)
        commitId = call("git rev-parse HEAD").strip()
        os.chdir(self.pathToBasicModels)
        execute("git add .")
        execute(f'git commit -m "basic_models {commitId}"')
        os.chdir(homePath)
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