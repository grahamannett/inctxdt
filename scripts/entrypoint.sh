#!/bin/bash

# use this to check entrypoint for time being
script_path=$1
other_args=${@:2}
echo "in docker - got conf path $script_path and other args $other_args"
echo -e "would run this conf $script_path which is this:\n$(cat $script_path)"

# run the script now
# bash $script_path
