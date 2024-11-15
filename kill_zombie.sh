#!/bin/bash

pkill -u $USER -f ns3
ps -eo pid,ppid,state,cmd | awk '$3 == "Z" {print $2}' | xargs kill -9