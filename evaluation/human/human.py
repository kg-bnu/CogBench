import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # ACL论文常用Arial
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42 

# 定义模型名称（更新为radarChart.py中的12个模型）
models = ['Glm-4.5', 'Glm-4', 'GPT-OoS-20B','Qwen3-30B-A3B', 'Qwen3-235B-A22B',  'DeepSeek-Chat',
          'CAT','GPT-5', 'GPT-5-Mini', 'Claude-4-Opus', 'Claude-4-Sonnet', 'Gemini-2.5-Flash', 'Gemini-2.5-Pro']

# 绘图数据已移除，需在使用前外部注入
data_a = []
data_b = []

# 定义颜色（浅色调）- 扩展到13个模型
bar_colors = ['#E0E0E0', '#F5DEB3', '#B0E0E6', '#FFB6C1', '#DDA0DD', '#FFE4E1',
              '#E0E0E0', '#F5DEB3', '#B0E0E6', '#FFB6C1', '#DDA0DD', '#FFE4E1', '#D8BFD8']

# 定义不同的填充图案（hatch patterns）- 扩展到13个模型
hatches = ['xxx', '|||', '---', '///', '\\\\\\', '+++', 
           '...', 'ooo', '***', 'xxx', '|||', '---', '///']

# 创建图形和子图（ACL单栏宽度，增加高度以容纳更大字体）
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))  # 稍微增大尺寸

# 定义分组位置（在第7-8个模型之间添加分隔线）
group_positions = [5.5,6.5]  # 在第7-8个模型之间添加分隔线

# 绘制两个子图
for i, (data, ax) in enumerate(zip([data_a, data_b], axes)):
    
    # 创建柱状图，每个柱子使用不同的颜色和图案
    bars = []
    for j, (model, value) in enumerate(zip(models, data)):
        bar = ax.bar(j, value, color=bar_colors[j], edgecolor='black', 
                     linewidth=0.6, hatch=hatches[j])
        bars.append(bar)
    
    # 设置y轴标题（增大字体）
    if i == 0:
        ax.set_ylabel('Consistency', fontsize=11)
    elif i == 1:
        ax.set_ylabel('Diversity', fontsize=12)
    
    # 设置y轴范围
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # 设置x轴标签
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90, ha='center', fontsize=11)
    
    # 添加分组分隔线
    for pos in group_positions:
        ax.axvline(x=pos, color='black', linestyle='--', linewidth=0.8)
    
    # 设置网格
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # 设置背景色
    ax.set_facecolor('white')
    
    # 设置y轴标签字体大小
    ax.tick_params(axis='y', labelsize=10)
    
    # 去掉坐标轴边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.24, wspace=0.3)  # 调整间距以适应单栏

# 先保存图表（必须在plt.show()之前）
fig.savefig('huamanEvaluation.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('huamanEvaluation.pdf', bbox_inches='tight', facecolor='white')


# 最后显示图表
plt.show()
