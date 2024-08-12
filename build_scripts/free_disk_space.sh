#!/usr/bin/env bash

# Based on a script from https://github.com/orgs/community/discussions/25678

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#
# The Azure provided machines typically have the following disk allocation:
# Total space: 85GB
# Allocated: 67 GB
# Free: 17 GB
# This script frees up 28 GB of disk space by deleting unneeded packages and 
# large directories.
# The Flink end to end tests download and generate more than 17 GB of files,
# causing unpredictable behavior and build failures.
#
echo "=============================================================================="
echo "Freeing up disk space on CI system"
echo "=============================================================================="

echo "Listing 100 largest packages (largest at bottom)"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100
df -h
echo "Removing large packages"
sudo apt-get remove -y '^ghc-8.*'
sudo apt-get remove -y '^dotnet-.*'
sudo apt-get remove -y '^llvm-.*'
sudo apt-get remove -y '^temurin-.*'
sudo apt-get remove -y 'php.*'
sudo apt-get remove -y azure-cli google-cloud-sdk google-cloud-cli hhvm google-chrome-stable firefox powershell mono-devel microsoft-edge-stable google-chrome-stable powershell linux-modules-6.5.0-1025-azure mysql-server-core-8.0 libllvm15 libllvm14
sudo apt-get autoremove -y
sudo apt-get clean
df -h
echo "Removing large directories"
# deleting 15GB
du -h -d 1 /opt/hostedtoolcache
rm -rf /opt/hostedtoolcache/Ruby
rm -rf /opt/hostedtoolcache/go
rm -rf /opt/hostedtoolcache/node
rm -rf /opt/hostedtoolcache/Java*
rm -rf /opt/hostedtoolcache/CodeQL
rm -rf /usr/share/dotnet/
df -h
