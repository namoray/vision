FILE=sam_vit_l_0b3195.pth
URL=https://dl.fbaipublicfiles.com/segment_anything/$FILE
if [ -f "$FILE" ]; then
    echo "$FILE exists, no need to download"
else 
    echo "$FILE does not exist, starting download."
    curl -O $URL
fi


# huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
