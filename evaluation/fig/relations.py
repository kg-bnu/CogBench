import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 定义指标（横坐标）
metrics = ['CAR', 'KAS', 'PAD', 'AS']

# 模型名称映射
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

open_source_models = {
    'Glm-4.5', 'Glm-4', 'Qwen3-30B-A3B', 
    'Qwen3-235B-A22B', 'GPT-OoS-20B', 'DeepSeek-Chat'
}

proprietary_models = {
    'GPT-5', 'GPT-5-Mini', 'Claude-4-Opus', 
    'Claude-4-Sonnet', 'Gemini-2.5-Flash', 'Gemini-2.5-Pro'
}

# 读取数据
metrics_dir = Path(__file__).parent.parent.parent / 'data' / 'metrics'

avg_data = {}  # 指标数据已移除，需在绘图前外部填充
excluded_models = ['gpt_5_nano_2025_08_07', 'gemini_2.5_flash_nothinking', 'gpt_4o_mini_2024_07_18']

line_color_palette = [
    '#F9A825', '#7CB342', '#E91E63', '#C2185B', '#00897B', '#FF5722',
    '#1976D2', '#388E3C', '#7B1FA2', '#C62828', '#F57C00', '#5E35B1',
]

# 单个图显示所有模型
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor('white')

x_positions = np.arange(len(metrics))

# 先绘制开源模型（实线）
color_idx = 0
open_handles = []
for model_name, values in avg_data.items():
    if model_name in open_source_models:
        y_values = values  # 直接使用四个指标的值
        color = line_color_palette[color_idx % len(line_color_palette)]
        line = ax.plot(x_positions, y_values, marker='o', linewidth=2.2, 
                linestyle='-', label=model_name, color=color, markersize=6, 
                markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=color)
        open_handles.extend(line)
        color_idx += 1

# 再绘制闭源模型（虚线区分）
for model_name, values in avg_data.items():
    if model_name in proprietary_models:
        y_values = values  # 直接使用四个指标的值
        color = line_color_palette[color_idx % len(line_color_palette)]
        ax.plot(x_positions, y_values, marker='s', linewidth=2.2,  # 方形标记区分
                linestyle='--', label=model_name, color=color, markersize=6,  # 虚线区分
                markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=color)
        color_idx += 1

ax.set_xlabel('Metrics', fontsize=15, weight='bold')
ax.set_ylabel('Score', fontsize=15, weight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(metrics, fontsize=15)
ax.set_ylim(0, 1.65)
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.set_yticklabels(['0.0', '0.4', '0.8', '1.2', '1.6'], fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

# 图例分两列，标注开源和闭源
legend = ax.legend(fontsize=9.5, frameon=True, framealpha=0.95, facecolor='white', 
          edgecolor='gray', loc='upper left', ncol=2, columnspacing=1.0,
          title='○ solid: Open Source LLMs    □ dash: Proprietary LLMs', title_fontsize=7.5)

plt.tight_layout()

plt.savefig('model_relations_comparison.pdf', bbox_inches='tight', dpi=300, 
           facecolor='white', format='pdf')
plt.savefig('model_relations_comparison.png', bbox_inches='tight', dpi=300, 
           facecolor='white')
print("折线图已保存为 model_relations_comparison.pdf 和 .png")
plt.show()
