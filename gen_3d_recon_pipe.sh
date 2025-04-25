#!/bin/bash


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name)
            name="$2"
            shift 2
            ;;
        --version)
            version="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done




for i in "$*"; do
    echo "$i"
done


source ~/.bashrc

# ======================================1=========================================
echo "[INFO] process No.1: Stablediffusion generation.."
echo "[INFO] Now getting args"

conda activate stablediffusion
if [$? != 0]; then
    echo "conda activate stablediffusion failed"
    exit 1
fi
py_ver=$(python --version)
echo "conda activate stablediffusion success, python version is $(py_ver)"


curr_path="/home/zhy01/Distill-Any-Depth/SD3.5_stream.py"
cd "$curr_path"

echo "cd to $curr_path"

# "-s",
# // "/home/zhy01/data/s1" this is for blender version
# $1 "/home/zhy01/桌面/feicuiwan_all",
# "--new_image_path",
# "/home/zhy01/Distill-Any-Depth/depth2"

base_path=$1
new_image_path=$2

python SD3.5_stream.py "-m" $base_path "new_image_path" $new_image_path


if [ $? -ne 0]; then
    echo "python SD3.5_stream.py failed"
    exit 1
fi
echo "[INFO] process No.1: Stablediffusion generation done.."








# =======================================2========================================

echo "[INFO] process No.2: Reconstrcuct with 3dgs..."
curr_path="/home/zhy01/gaussian-splatting/train.py"
cd $curr_path
echo "cd to $curr_path"

conda activate gaussian_splatting
if [$? -ne 0]; then
    echo ""conda activate gaussian_splatting failed"
    exit 1
fi
py_ver=$(python --version)
echo "conda activate gaussian_splatting success, python version is $(py_ver)"


source_path=$3
new_image_path=$4


python train.py "-m" $source_path "--new_image_path" $new_image_path


if [ $? -ne 0]; then
    echo "python train.py failed"
    exit 1
fi

echo "[INFO] process No.2: generation 3dgs  done.."





# ====================================3 render===================================


