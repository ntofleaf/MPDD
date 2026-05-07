log_file=$1

# python tools/analysis_tools/analyze_logs.py plot_curve $log_file --out pcc.jpg --legend pcc ccc --keys accuracy/pcc accuracy/ccc

python tools/analysis_tools/analyze_logs.py plot_curve $log_file --out tmp/loss.jpg --legend loss --keys loss

python tools/analysis_tools/analyze_logs.py plot_curve $log_file --out tmp/acc.jpg --legend acc --keys accuracy/cls_acc
python tools/analysis_tools/analyze_logs.py plot_curve $log_file --out tmp/rmse.jpg --legend rmse --keys accuracy/rmse
