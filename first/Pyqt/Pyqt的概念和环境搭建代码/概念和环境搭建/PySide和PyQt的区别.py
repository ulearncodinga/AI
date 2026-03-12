# 什么是Pyside
'''
PySide，最初是由是由Nokia公司开发，现在由 Qt 公司维护，它是 Qt 框架的另一个 Python 绑定。
PySide2 是针对 Qt 5 的版本，而 PySide6 是针对最新 Qt 6 版本的 Python 绑定。PySide是跨平台的图
形使用界面框架Qt的Python版本。提供和PyQt类似的功能，并兼容API。但与PyQt不同处为使用LGPL
许可。他是一个专门用于创建GUI的库，可以在Python中使用，也可以在不同的操作系统上使用
（linux&windows）。此外大多数的DCC软件都使用它，比如Maya、houdini和Blender等。Pyside基
于QT的C++框架。
'''
#许可协议
'''
PyQt5 提供了两种许可选项：GPL 和商业许可。使用 GPL 许可证时，你的应用程序必须开源；如
果选择商业许可，则需要付费。
PySide采用LGPL（GNU Lesser General Public License）开源协议，LGPL是一个较为宽松的开源
协议，与 GPL 相比，它允许商业软件在不公开源代码的情况下使用 LGPL 许可的库。这意味着你可
以使用 PySide 来开发商业应用程序，而不必开源你的整个应用程序，只需确保对 PySide 本身所做
的任何修改是开源的。
'''
#开发和维护
'''
PySide 由 The Qt Company 维护，它是 Qt 框架的官方 Python 绑定。
PyQt5 由 Riverbank Computing 维护，它是一个第三方解决方案。
'''
#API兼容性
'''
PySide 和 PyQt5的 API 非常相似，但并不完全相同。在某些情况下，类名、方法名或参数可能会
有所不同。因此，从一个库迁移到另一个库可能需要一些代码修改
'''
"""性能"""
#在性能方面，PySide 和 PyQt5 通常非常相似，因为它们都依赖于相同的 Qt 库。但是，具体性能
# 可能会因实现细节和优化而有所不同。