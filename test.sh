# test.sh
# !/bin/sh
pssh -h $PBS_NODEFILE -i "if [ ! -d \"/home/sTest/test\" ];then mkdir -p \"/home/sTest/test\"; fi" 1>&2  # 这里有一个if语句，是保证mkdir命令执行不会报错，也就是如果文件夹不存在则创建路径。文件夹路径就是/home/sTest/test。
pscp -h $PBS_NODEFILE /home/sTest/test/hello /home/sTest/test 1>&2  # 第一个文件路径是你的可执行文件在master节点的路径，第二个文件夹路径是你希望把文件发送到其他计算节点的具体文件夹，应与当前节点可执行文件所在文件夹路径保持一致。
/home/sTest/test/hello  # 执行文件
# 使用本脚本前，先把第三四五行#号后的注释以及本行注释删掉。