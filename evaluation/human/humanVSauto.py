import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # ACL论文常用Arial
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

# 根据图片描述的数据
models = ['Glm-4.5', 'Glm-4', 'GPT-OoS-20B','Qwen3-30B-A3B', 'Qwen3-235B-A22B',  'DeepSeek-Chat',
          'CAT','GPT-5', 'GPT-5-Mini', 'Claude-4-Opus', 'Claude-4-Sonnet', 'Gemini-2.5-Flash', 'Gemini-2.5-Pro']

# 模型名称到JSON文件名的反向映射
model_to_file = {
    'GPT-OoS-20B': 'gpt_oss_20b_no_limit',
    'Qwen3-30B-A3B': 'qwen3_30b_a3b',
    'Qwen3-235B-A22B': 'qwen3_235b_a22b_instruct_2507',
    'GPT-5': 'gpt_5_2025_08_07',
    'GPT-5-Mini': 'gpt_5_mini_2025_08_07',
    'Claude-4-Opus': 'claude_opus_4_20250514',
    'Claude-4-Sonnet': 'claude_sonnet_4_20250514',
    'Gemini-2.5-Flash': 'gemini_2.5_flash',
    'Gemini-2.5-Pro': 'gemini_2.5_pro_thinking_156',
    'DeepSeek-Chat': 'deepseek_chat',
    'Glm-4.5': 'glm_4.5',
    'Glm-4': 'glm_4'
}

# 绘图所需的自动与人工评分数据已移除，需在使用前外部注入
automatic_scores = []
human_scores_raw_1 = []
human_scores_raw_2 = []
human_scores_raw = []
human_scores = []

# 创建数据矩阵
data_matrix = np.array([automatic_scores, human_scores])

if data_matrix.size == 0:
    raise RuntimeError("Evaluation score data has been removed; please populate automatic_scores and human_scores.")

# 创建图形
fig, ax = plt.subplots(figsize=(14, 5))

# 创建热力图，使用与图片相同的颜色方案，范围调整为0-100
im = ax.imshow(data_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)

# 设置刻度标签
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=16)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Automatic Score', 'Human Score'], fontsize=16)

# 在热力图中添加数值
for i in range(len(models)):
    for j in range(2):
        # 根据背景颜色调整文字颜色
        if data_matrix[j, i] > 50:
            text_color = 'white'
        else:
            text_color = 'black'
        
        text = ax.text(i, j, f'{data_matrix[j, i]:.1f}', 
                      ha="center", va="center", color=text_color, 
                      fontweight='bold', fontsize=16)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Score (Normalized to 0-100)', rotation=270, labelpad=20, fontsize=16)

# 设置标题
# plt.title('Model Performance Comparison: Automatic vs Human Scores (Normalized)', 
#           fontsize=14, fontweight='bold', pad=20)

# 添加网格线
ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=1)

# 调整布局
plt.tight_layout()


# 保存图表
plt.savefig('humanVSauto.png', dpi=300, bbox_inches='tight')
plt.savefig('humanVSauto.pdf', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
