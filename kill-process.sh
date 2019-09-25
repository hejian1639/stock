#!/bin/bash

set -e

python json-read.py

NAME=ssserver
echo $NAME
ID=`ps -ef | grep "$NAME" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
echo $ID

echo "---------------"
for id in $ID
do
    kill $id
    echo "killed $id"
done
echo "---------------"

nohup ssserver -c config.json &

echo "port changed"
