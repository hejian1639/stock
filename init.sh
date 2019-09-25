#!/bin/bash

set -e

apt-get update
apt-get install shadowsocks -y

chmod +x kill-process.sh
./kill-process.sh

echo 'initial finished'