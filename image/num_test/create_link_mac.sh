#! /bin/bash

function find_dir(){
    i=0
    for file in ` ls $1 `
    do
        if [ -d $1"/"$file ]
        then
            read_image $1"/"$file $i;
            i=`expr $i + 1`;
        fi
    done
}

function read_image(){
    for file in ` ls $1'/' `
    do
        echo $(greadlink -f $1'/'$file' '$2)>> "../../config/test.txt"
    done
}

> ../../config/test.txt
find_dir $(greadlink -f .)
