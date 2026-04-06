import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use('science')
import numpy as np
import json
import os
from pathlib import Path

# 设置字体为现代无衬线字体（类似图中的字体）
plt.rcParams['font.sans-serif'] = ['Arial']  # ACL论文常用Arial
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10  # 基础字体大小

# 0. 全局定义指标
metrics = ['ACC', 'CAR', 'KAS', 'PAD', 'AS']
N = len(metrics)
# 计算角度并闭合
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 1. 定义模型名称映射（JSON文件名 -> 显示名称）
model_name_mapping = {
    'gpt_oss_20b_no_limit': 'GPT-OoS-20B',
    'qwen3_30b_a3b': 'Qwen3-30B-A3B',
    'qwen3_235b_a22b_instruct_2507': 'Qwen3-235B-A22B',
    'gpt_5_2025_08_07': 'GPT-5',
    'gpt_5_mini_2025_08_07': 'GPT-5-Mini',
    'claude_opus_4_20250514': 'Claude-4-Opus',
    'claude_sonnet_4_20250514': 'Claude-4-Sonnet',
    'gemini_2.5_flash': 'Gemini-2.5-Flash',
    'gemini_2.5_pro_thinking_156': 'Gemini-2.5-Pro',
    'deepseek_chat': 'DeepSeek-Chat',
    'glm_4.5': 'Glm-4.5',
    'glm_4': 'Glm-4'
}

# 定义开源模型和闭源模型分类
open_source_models = {
    'Glm-4.5',
    'Glm-4',
    'Qwen3-30B-A3B',
    'Qwen3-235B-A22B',
    'GPT-OoS-20B',
    'DeepSeek-Chat'
}

proprietary_models = {
    'GPT-5',
    'GPT-5-Mini',
    'Claude-4-Opus',
    'Claude-4-Sonnet',
    'Gemini-2.5-Flash',
    'Gemini-2.5-Pro'
}

# 2. 读取所有JSON文件并提取数据
metrics_dir = Path(__file__).parent.parent.parent / 'data' / 'metrics'

direct_data = {}
grade_aware_data = {}
knowledge_aware_data = {}
avg_data = {}

# 要排除的模型文件名
excluded_models = ['gpt_5_nano_2025_08_07', 'gemini_2.5_flash_nothinking', 'gpt_4o_mini_2024_07_18']

# 指标数据已移除，需在绘制前外部填充以上数据字典

# 3. 过滤数据函数
def filter_data_by_models(data_dict, model_set):
    """根据模型集合过滤数据"""
    return {k: v for k, v in data_dict.items() if k in model_set}

# 4. 绘制雷达图的函数
def plot_radar_charts(data_dicts, title_suffix, filename_suffix):
    """绘制一组雷达图"""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')  # 白色背景
    
    datasets = [
        (data_dicts['direct'], "Direct"),
        (data_dicts['grade'], "Grade-Aware"),
        (data_dicts['knowledge'], "Knowledge-Aware"),
        (data_dicts['avg'], "Avg")
    ]
    
    # 循环绘制四个雷达图
    for idx, (data, title) in enumerate(datasets, 1):
        ax = fig.add_subplot(1, 4, idx, polar=True)
        
        # 设置白色背景
        ax.set_facecolor('white')
        
        # 关闭默认的网格线（避免显示径向网格线）
        ax.grid(False)
        
        # 设置轴线和刻度为黑色
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('black')
        ax.tick_params(colors='black', labelsize=8)
        
        # 定义配色方案（匹配图片中的浅色调）
        color_palette = [
            '#e7e8fc',  # 黄色（Yellowish）
            '#e7e8fc',  # 浅绿色（Light Green）
            '#e7e8fc',  # 浅粉色（Light Pink）
            '#e7e8fc',  # 红粉色（Reddish-Pink）
            '#e7e8fc',  # 浅蓝/青色（Light Blue/Teal）
            '#e7e8fc',  # 浅橙/桃色（Light Orange/Peach）
            '#e7e8fc',  # 浅蓝色（Light Blue）
            '#e7e8fc',  # 浅绿/青色（Light Green/Teal）
            '#e7e8fc',  # 浅紫色（Light Purple）
            '#e7e8fc',  # 红/橙色（Red/Orange）
            '#e7e8fc',  # 浅橙色（备用）
            '#e7e8fc',  # 浅紫蓝（备用）
        ]
        
        # 对应的深色版本用于线条
# 对应的深色版本用于线条
        line_color_palette = [
            '#F9A825',  # 深黄色（Yellowish）
            '#7CB342',  # 深绿色（Light Green）
            '#E91E63',  # 深粉色（Light Pink）
            '#C2185B',  # 深红粉色（Reddish-Pink）
            '#00897B',  # 深青色（Light Blue/Teal）
            '#FF5722',  # 深橙色（Light Orange/Peach）
            '#1976D2',  # 深蓝色（Light Blue）
            '#388E3C',  # 深绿/青色（Light Green/Teal）
            '#7B1FA2',  # 深紫色（Light Purple）
            '#C62828',  # 深红色（Red/Orange）
            '#F57C00',  # 深橙色（备用）
            '#5E35B1',  # 深紫蓝（备用）类似 #6d62e2
        ]
        
        # 循环绘制每个模型
        for i, (model_name, values) in enumerate(data.items()):
            plot_values = values + values[:1]  # 闭合数据
            fill_color = color_palette[i % len(color_palette)]
            line_color = line_color_palette[i % len(line_color_palette)]
            
            # 绘制填充区域（浅色，匹配图片样式）
            ax.fill(angles, plot_values, color=fill_color, alpha=0.3, linewidth=0)
            
            # 绘制数据线（粗线，使用深色）
            ax.plot(angles, plot_values, color=line_color, linewidth=1.5, 
                   label=model_name, marker='o', markersize=4, markerfacecolor='white', 
                   markeredgewidth=1.5, markeredgecolor=line_color)
        
        # 设置标签和刻度（黑色）
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=13, color='black', weight='normal')
        ax.set_rlim(0, 1.6)
        ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
        ax.set_yticklabels(["0", "0.4", "0.8", "1.2", "1.6"], color='black', size=8)
        
        # 设置径向网格线（先设置刻度位置，类似R代码的caxislabels）
        ax.set_rgrids([0, 0.4, 0.8, 1.2, 1.6], labels=None, angle=0)
        
        # 移除默认的圆形网格线
        for line in ax.yaxis.get_gridlines():
            line.set_visible(False)
        
        # 手动绘制五边形的轴线（从中心到每个顶点）
        pentagon_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        for angle in pentagon_angles:
            ax.plot([angle, angle], [0, 1.6], color='black', linewidth=0.8, 
                   alpha=0.6, linestyle='-', zorder=0)
        
        # 手动绘制五边形网格线（同心五边形）
        r_values = [0.4,0.8, 1.2, 1.6]  # 对应刻度值，跳过0（中心点）
        pentagon_angles_closed = list(pentagon_angles) + [pentagon_angles[0]]  # 闭合形成五边形
        
        for r_val in r_values:
            pentagon_r = [r_val] * len(pentagon_angles_closed)
            ax.plot(pentagon_angles_closed, pentagon_r, color='black', linewidth=0.8, 
                   alpha=0.6, linestyle='-', zorder=0)
        
        # 添加标题
        ax.set_title(title, size=16, weight='bold', pad=15, color='black')
        
        # 只在最后一个子图添加图例
        if idx == 4:
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=16, 
                    frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    filename = f"radar_charts_{filename_suffix}.pdf"
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    print(f"{title_suffix}雷达图已保存为 {filename}")
    plt.show()

# 5. 绘制开源模型的雷达图
open_source_data = {
    'direct': filter_data_by_models(direct_data, open_source_models),
    'grade': filter_data_by_models(grade_aware_data, open_source_models),
    'knowledge': filter_data_by_models(knowledge_aware_data, open_source_models),
    'avg': filter_data_by_models(avg_data, open_source_models)
}

plot_radar_charts(open_source_data, "开源模型", "open_source")

# 6. 绘制闭源模型的雷达图
proprietary_data = {
    'direct': filter_data_by_models(direct_data, proprietary_models),
    'grade': filter_data_by_models(grade_aware_data, proprietary_models),
    'knowledge': filter_data_by_models(knowledge_aware_data, proprietary_models),
    'avg': filter_data_by_models(avg_data, proprietary_models)
}

plot_radar_charts(proprietary_data, "闭源模型", "proprietary")
