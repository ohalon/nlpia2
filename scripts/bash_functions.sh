#!/usr/bin/env bash
echo "Creating functions like *workon()* in $HOME/bin/bash_functions.sh: $0..."
function workon() {
    if [ -z $1 ]
    then
        echo "Usage: workon DIR_OR_ENV_NAME"
    else
        # echo "Time to workon..."
        conda_sufs=( "" "env" "_env" "37" "38" "36" "27" )
        conda_dirs=( "$HOME/opt/anaconda3" "$HOME/anaconda3" "$HOME/miniconda" "$HOME/anaconda" "/opt/anaconda3" )
        for conda_suf in "${conda_sufs[@]}"
        do
            for conda_dir in "${conda_dirs[@]}"
            do
                env_name="$1$conda_suf"
                full_path="$conda_dir/envs/$env_name"
                # echo $full_path
                if [ -d "$full_path" ]
                then
                    echo "FOUND CONDA ENV: $full_path"
                    conda activate "$env_name"
                    break
                fi
                full_path="$conda_dir/envs/$env_name"
                # echo $full_path
                if [ -d "$full_path" ]
                then
                    break
                fi
            done
            if [ -d "$full_path" ]
            then
                echo "FOUND CONDA ENV: $full_path"
                conda activate $env_name
                break
            fi
        done
        code_dirs=( "$HOME/code/tangibleai" "$HOME/code/entolabs" "$HOME/code/teaching/ucsd" "$HOME/code/teaching" "$HOME/code/chatbot" "$HOME/code" "$HOME/src" )  # "$HOME/code/mindcurrent"
        for base_dir in "${code_dirs[@]}"
        do
            full_path="$base_dir/$1"
            # echo $full_path
            if [ -d "$full_path" ]
            then
                # echo "FOUND DIR: $full_path"
                # cd "$full_path"
                break
            else
                echo "! [ -d $full_path"
            fi
        done
        if [ -d "$full_path" ]
        then
            echo "FOUND DIR: $full_path"
            cd "$full_path"
        else
            if [ -d "$base_dir" ]
            then
                echo ""
            fi
        fi
    fi
}
