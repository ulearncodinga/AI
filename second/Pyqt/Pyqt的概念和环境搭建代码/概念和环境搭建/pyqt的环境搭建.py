"""
3. PyQT的安装
PyQT的安装包含两部分：PyQT的库的安装，PyQT的工具。
3.1 PyQT库的介绍与安装
要安装 PyQt，你可以使用 pip ，这是 Python 的包管理器。首先，确保你已经安装了 Python。然后，
打开终端或命令提示符，并运行以下命令：
这会安装 PyQt5 库及其相关的依赖项。如果下载较慢的话，可以更换国内镜像源进行下载：
请注意，上述命令是用于 PyQt5 的，如果你需要安装 PyQt6，可以将命令改为：
选择 PyQt5 还是 PyQt6 取决于你的项目需求和 Qt 版本的兼容性。一般来说，如果你要使用 Qt 5.x，就
选择 PyQt5；如果你要使用 Qt 6.x，就选择 PyQt6。
注意：PyQt5支持Python2.x、Python3.5.x-3.9.x的版本，Python3.10及以上版本可能并不兼容
PyQt5
安装完成后，你就可以在 Python 中使用 PyQt 来开发桌面应用程序了。在你的代码中，你需要导入
PyQt 模块，例如：
这里示范的是导入 PyQt5 的部分模块。如果你使用 PyQt6，对应的导入语句会有所不同。
请注意，有时候 PyQt 的安装可能会依赖于 Qt 的库，因此你可能需要确保已经安装了 Qt。在某些系统
上，你可能需要安装 Qt 的开发工具包，例如在 Ubuntu 上可以使用以下命令：
这样，你就可以开始使用 PyQt 来构建强大的桌面应用程序了。
3.2 PyQT工具的介绍与安装
PyQt Tools 是 PyQt 框架提供的一组工具，主要用于辅助 PyQt 应用程序的开发和设计，下载命令如下
（如果下载速度慢，可以添加国内镜像源）：
以下是 PyQt Tools 中一些常见的工具：
pip install PyQt5==5.15.2
pip install PyQt5==5.15.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install PyQt6
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
# 你的代码继续...
sudo apt-get install qt5-default
pip install PyQt5-tools==5.15.2.3.1
1. Qt Designer： 这是一个可视化的用户界面设计器，允许开发者通过拖放方式设计和布局应用程序
的界面。设计好的界面可以保存为.ui 文件，然后可以使用 pyuic 工具将其转换为相应的 Python
代码，以便在 PyQt 应用程序中使用。
2. PyUIC： 这是一个用于将 Qt Designer 设计的 .ui 文件转换为对应的 Python 代码的命令行工具。
通过这个工具，可以将可视化设计的用户界面集成到 PyQt 项目中。
3. PyRCC： 用于将 Qt 资源文件 (.qrc) 编译成 Python 模块的工具。Qt 资源文件可以包含图像、文本
文件等，通过编译可以将这些资源文件嵌入到 PyQt 应用程序中，方便统一管理。
4. Pylupdate： 用于更新 Qt 项目的翻译文件 (.ts)。这在多语言支持的应用程序中很有用，可以提供
不同语言的界面翻译。
一般情况下，需要使用的是前三个。
3.3 工具的导入
可以直接在Pycharm使用这些工具，在 PyCharm 中导入 PyQt 工具通常是通过设置外部工具（External
Tools）来完成的。下面是一些基本步骤，以在 PyCharm 中导入 PyQt 工具：
在 PyCharm 中导入 PyQt 工具通常是通过设置外部工具（External Tools）来完成的。下面是一些基本
步骤，以在 PyCharm 中导入 PyQt 工具：
1. 打开 PyCharm： 打开你的 PyCharm IDE。
2. 进入设置（Settings）： 在顶部菜单栏中，选择 "File" -> "Settings"（在 macOS 上是 "PyCharm"
-> "Preferences"）。
3. 选择 External Tools： 在设置窗口左侧的导航栏中，展开 "Tools" 节点，并选择 "External
Tools"。
4. 添加外部工具： 在右侧窗格中，点击 "+"（添加）按钮，然后选择 "Program"。
5. 填写外部工具信息： 在弹出的对话框中，填写以下信息：
Name（名称）: 工具的名称，例如 "pyuic" 或 "Qt Designer"。
Group（组）: 可选，可以将工具分组，这样可以更好地组织工具列表。
Description（描述）: 工具的描述，用于说明工具的作用。
Program（程序）: 工具的可执行文件路径。例如，对于 pyuic，你需要指定它的完整路径，
通常是在虚拟环境中的 Scripts （Windows）或 bin （Linux/Mac）目录中。
Arguments（参数）: 传递给工具的参数。对于 pyuic，这可能是一些选项和输入文件的路
径。
Working Directory（工作目录）: 工具运行时的工作目录，通常可以设置为你的项目根目
录。
Qt Designer，图形化布局工具：
 Lib\site-packages\qt5_applications\Qt\bin\designer.exe （程序路径）
 $FileDir$ (当前文件夹)
 $ProjectFileDir$ （绝对路径）
PyUIC，图形界面转py：
 Scripts\pyuic5.exe （程序路径）
 $FileName$ -o $FileNameWithoutExtension$.py （当前文件转为.py文件）
 $FileDir$
Pyrcc ，资源转py：
 Scripts\pyrcc5.exe （程序路径）
 $FileName$ -o $FileNameWithoutExtension$_rc.py （当前文件转为.py文件）
6. 保存设置： 点击 "OK" 保存外部工具设置。
7. 运行外部工具： 在 PyCharm 主界面中，选择 "Tools" 菜单，然后选择你添加的外部工具。你也可
以使用快捷键或在右键菜单中找到。
4. PyQT5的常用模块介绍
4.1 QtCore模块
包含了Qt的核心非GUI功能，如事件循环、定时器、线程、文件和文件夹操作、时间和日期等。这个模
块是PyQt5的基础。
4.2 QtGui模块
包含了Qt的GUI相关功能，如窗口、窗口部件、绘图、颜色、字体、图片等。这个模块提供了创建和管
理用户界面的基本工具。
4.3 QtWidgets模块
是PyQt5中最常用和最重要的模块，提供了一系列用户界面控件，如按钮、文本框、标签、表格、菜
单、滚动条等。开发者可以利用这些控件快速构建用户界面。
4.4 QtMultimedia模块
用于处理多媒体相关的功能，如音频、视频、摄像头等。这个模块提供了访问和处理多媒体资源的接
口。
4.5 QtNetwork模块
用于处理网络通信相关的功能，如TCP/IP、UDP、HTTP等。这个模块提供了与网络相关的类和方法，方
便开发者进行网络编程。
4.6 QtSql模块
用于数据库操作，支持多种数据库系统，如SQLite、MySQL、Oracle等。这个模块提供了访问和操作数
据库的功能。
4.7 QtWebEngine模块
用于在应用程序中嵌入Web内容，如网页浏览器、HTML5应用等。
4.8 QtWebEngineWidgets模块
提供了WebEngine模块的Qt Widgets集成，用于在Qt应用程序中显示和控制Web内容。
4.9 其它
除了以上的核心模块外，PyQt5还包含了其他一些辅助模块，如QtPrintSupport（打印支持）、QtSvg
（矢量图形支持）、QtTest（单元测试支持）等。这些模块提供了额外的功能，方便开发者进行更加丰
富和多样化的应用程序开发。
"""
