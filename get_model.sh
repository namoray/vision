FILE=sam_vit_l_0b3195.pth
URL=https://dl.fbaipublicfiles.com/segment_anything/$FILE
if [ -f "$FILE" ]; then
    echo "$FILE exists, no need to download"
else 
    echo "$FILE does not exist, starting download."
    curl -O $URL
fi