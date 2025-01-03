import subprocess

# 打开并读取requirements.txt文件
with open('/home/gy237/project/requirement.txt', 'r') as file:
    lines = file.readlines()

# 按顺序逐行安装包
for line in lines:
    package = line.strip()
    if package:
        print(f"Installing {package}")
        subprocess.check_call(['pip', 'install', package])