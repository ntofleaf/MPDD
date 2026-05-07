#!/usr/bin/env bash
# Usage: bash train.sh --config <config_file> --dir <directory> --times <times>
CONFIG_FILE="configs/video_level/vat_anchor_freeform.py"
DIRECTORY="work_dirs/vat_anchor_freeform"
TIMES=10

# 处理关键字参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dir)
            DIRECTORY="$2"
            shift 2
            ;;
        --times)
            TIMES="$2"
            shift 2
            ;;
        *)  # 未知选项
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

for ((i=0; i<TIMES; i++)); do
    echo "Start training $i-th time..."
    echo "Config file: $CONFIG_FILE ; Directory: $DIRECTORY"
    python tools/train.py $CONFIG_FILE --work-dir $DIRECTORY
    sleep 3
done
