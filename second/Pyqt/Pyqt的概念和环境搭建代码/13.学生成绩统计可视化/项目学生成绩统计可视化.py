import numpy as np#导入Numpy库
import pandas as pd#导入Pandas库
import matplotlib.pyplot as plt#导入Matplotlib模块
from matplotlib.backends.backend_pdf import PdfPages#用于绘制PDF文件

# ---------------------- 1. 解决中文乱码（关键） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']#解决中文显示问题
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ---------------------- 2. 数据读取与预处理（适配你的Excel列名） ----------------------
df = pd.read_excel('./source.xlsx')#读取excel表
print("你的Excel文件实际列名：")
print(df.columns.tolist())#获取所有列名
print("\n原始数据预览：")
print(df.head())#这个默认打印前5行数据,预览用

# 填充考试/出勤分数的缺失值为0（仅处理核心数值列）
df['exam'] = df['exam'].fillna(0)
df['attendance'] = df['attendance'].fillna(0)

# ---------------------- 3. 成绩计算与通过判断 ----------------------
# 计算最终成绩（考试70% + 出勤30%，四舍五入取整）
df['finally'] = np.round(df['exam'] * 0.7 + df['attendance'] * 0.3)
# 中文标签判断是否通过（更直观）
df['pass'] = df['finally'].apply(lambda x: '通过' if x >= 60 else '未通过')

print("\n数据处理完成后预览（含最终成绩和通过情况）：")
print(df[['name', 'exam', 'attendance', 'finally', 'pass']].head())

# ---------------------- 4. 功能1：筛选80分及以上学生（显示姓名+学号） ----------------------
score_threshold = 80
high_score_students = df[df['finally'] >= score_threshold].copy()
high_score_students['序号'] = range(1, len(high_score_students) + 1)
high_score_display = high_score_students[['序号', 'stuid', 'name', 'finally', 'pass']]

print(f"\n{score_threshold}分及以上的学生（共{len(high_score_students)}人，前10名）：")
print(high_score_display.head(10))

# ---------------------- 5. 功能2：成绩排名（降序，显示姓名） ----------------------
df_ranked = df.sort_values(by='finally', ascending=False).copy()
df_ranked['排名'] = range(1, len(df_ranked) + 1)
ranked_display = df_ranked[['排名', 'stuid', 'name', 'finally', 'pass']]

print("\n按最终成绩排名（前10名）：")
print(ranked_display.head(10))

# ---------------------- 6. 功能3：班级成绩整体统计（含学生姓名） ----------------------
total_count = len(df)
class_average = df['finally'].mean()
class_max = df['finally'].max()
class_min = df['finally'].min()
pass_count = len(df[df['pass'] == '通过'])
pass_rate = (pass_count / total_count) * 100

# 获取最高分/最低分学生姓名（处理可能的多人同分情况）
max_name = '、'.join(df[df['finally'] == class_max]['name'].values)
min_name = '、'.join(df[df['finally'] == class_min]['name'].values)

print(f"\n班级成绩整体统计：")
print(f"• 参与统计人数：{total_count} 人")
print(f"• 平均分：{class_average:.2f} 分")
print(f"• 最高分：{class_max} 分（{max_name}）")
print(f"• 最低分：{class_min} 分（{min_name}）")
print(f"• 通过率：{pass_rate:.1f}% （{pass_count}人通过 / {total_count}人参考）")

# ---------------------- 7. 可视化：直方图+饼图合并（弹窗显示+保存PDF） ----------------------
# 创建1行2列子图，控制尺寸避免拥挤
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ---------------------- 左子图：最终成绩分布直方图 ----------------------
bins = np.arange(0, 111, 10)  # 分数区间：0-10,10-20,...,100-110
hist, bin_edges = np.histogram(df['finally'], bins=bins)
bar_width = bin_edges[1] - bin_edges[0]  # 条形宽度=10分

# 绘制直方图
bars = ax1.bar(
    bin_edges[:-1],  # x轴：区间起点
    hist,            # y轴：区间人数
    width=bar_width,
    align='edge',    # 条形左边缘对齐区间起点
    color='#2E86AB', # 蓝色系（美观且清晰）
    alpha=0.8,
    edgecolor='black'
)

# 给每个条形添加人数标签
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax1.text(
            bar.get_x() + bar_width/2,
            height + 0.2,
            str(int(height)),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

# 直方图样式优化
ax1.set_title('学生最终成绩分布直方图', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('分数区间', fontsize=12)
ax1.set_ylabel('学生人数', fontsize=12)
# x轴刻度标签改为“0-10”“10-20”格式，更易读
ax1.set_xticks(bin_edges[:-1])
ax1.set_xticklabels([f'{int(bin)}-{int(bin)+10}' for bin in bin_edges[:-1]])
ax1.grid(axis='y', linestyle='--', alpha=0.5)  # 水平网格线
# 添加及格线（60分）和图例
ax1.axvline(x=60, color='red', linestyle='--', linewidth=2, label='及格线（60分）')
ax1.legend(fontsize=10)

# ---------------------- 右子图：通过情况饼图（修复格式错误） ----------------------
pass_stats = df['pass'].value_counts()  # 统计“通过”“未通过”人数
colors = ['#A23B72', '#F18F01']  # 紫红色（通过）、橙色（未通过）
explode = (0.05, 0)  # 突出“通过”部分

# 绘制饼图（修复autotexts格式：用lambda函数动态匹配数值）
wedges, texts, autotexts = ax2.pie(
    pass_stats.values,
    labels=pass_stats.index,
    autopct=lambda pct: f'{pct:.1f}%\n（{int(pct/100*total_count)}人）',  # 动态计算人数
    explode=explode,
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11}
)

# 优化饼图文字样式
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 饼图标题
ax2.set_title('学生通过情况比例图', fontsize=14, fontweight='bold', pad=20)

# ---------------------- 保存PDF+弹窗显示图表 ----------------------
# 调整子图间距，避免标签重叠
plt.tight_layout()

# 保存到PDF
with PdfPages('student_scores_analysis.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')  # 避免边缘裁剪
    print("\n 图表已保存到 'student_scores_analysis.pdf'")

# 弹窗显示图表（关键：运行后自动弹出）
plt.show()

print("\n 学生成绩统计与可视化全部完成！")