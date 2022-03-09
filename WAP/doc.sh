#!/bin/bash

if [ $# -eq 1 ]; then
    if [ $1 == "--install" ] 
    then
        npm install jsdoc
    fi
fi

jsdoc -c conf.json tree.mjs