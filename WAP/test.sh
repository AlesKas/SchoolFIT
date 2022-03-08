#!/bin/bash

if [ $# -eq 1 ]; then
    if [ $1 == "install" ] 
    then
        npm install mocha
        npm install chai
    fi
fi

npm test