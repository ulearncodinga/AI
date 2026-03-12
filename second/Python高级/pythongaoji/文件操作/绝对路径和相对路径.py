'''
 文件路径
'''

#绝对路径
"""
绝对路径是从文件系统的根目录开始的一条完整路径,它指明了从根目录到达目标文件或目录的具体步骤
特点:
不依赖于当前工作目录
在不同的用户或程序之间具有一致性
提供了文件或目录的完整位置信息
windows系统中绝对路径大概长这样
C:\Users\John\Documents\report.txt
linux系统:
/home/john/Documents/report.txt
"""
#相对路径
"""
是指相对于某个起始点(通常是当前工作目录)到达目标文件或目录路径
特点:
取决于当前工作目录的位置
更加灵活,但可能以为上下文变化而变化
适用于在同一目录层级或者附近层级内的文件访问
在Windows系统中，
如果当前工作目录是C:\Users\John\Documents
，那么我想去找同目录下的example.txt的话，就是.\example.txt，如果上级目录下的example.txt的话，就是..\example.txt。
在Linux/Unix中，
如果当前工作目录是/home/john/Documents，那么我想去找同目录下的example.txt的话，就是./example.txt，
如果上级目录下的example.txt的话，就是../example.txt。
"""

"""
在window是系统中用绝对路径,需要对斜杠进行转义用\对\进行转义 \\
"""



