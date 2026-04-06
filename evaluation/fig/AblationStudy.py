import matplotlib.pyplot as plt
import numpy as np
import json

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11  # 基础字体稍大

# 指标数据已移除，需在绘图前填充models_data
models_data = {}

# 处理数据 - 直接使用原始值（不再乘以1.6）
processed_data = {}
for model_name, values in models_data.items():
    processed_data[model_name] = [
        values['acc'],
        values['car'],
        values['kas'],
        values['pad'],
        values['as']
    ]

# 只使用4个指标（去掉ACC）
metrics = ['CAR', 'KAS', 'PAD', 'AS']

# 打印数据对比
print("=" * 60)
print("模型平均指标对比（原始数据）")
print("=" * 60)
for model_name, values in processed_data.items():
    print(f"\n{model_name}:")
    print(f"  CAR: {values[1]:.4f}")
    print(f"  KAS: {values[2]:.4f}")
    print(f"  PAD: {values[3]:.4f}")
    print(f"  AS: {values[4]:.4f}")
print("=" * 60)

if not processed_data:
    raise RuntimeError("Ablation metrics data has been removed; please populate models_data before plotting.")

# 创建2行2列的子图布局 - 优化为单栏尺寸
fig, axes = plt.subplots(2, 2, figsize=(7.5, 9))
fig.patch.set_facecolor('white')
axes = axes.flatten()

model_names = list(processed_data.keys())
x_pos = np.arange(len(model_names))

# 定义不同的填充图案（hatch patterns）
hatches = ['xxx', '|||', '---', '///', '\\\\\\', '+++']

# 定义颜色（浅色调）
bar_colors = ['#E0E0E0', '#F5DEB3', '#B0E0E6', '#FFB6C1', '#DDA0DD', '#FFE4E1']

# 指标索引映射（跳过ACC，从CAR开始）
metric_indices = [1, 2, 3, 4]  # CAR, KAS, PAD, AS

# 找到Teacher和GPT-5M的索引位置
teacher_idx = None
gpt5m_idx = None
for i, name in enumerate(model_names):
    if 'Cognition-aware Teacher' == name:
        teacher_idx = i
    elif 'GPT-5-Mini' in name:
        gpt5m_idx = i

# 为每个指标绘制一个子图
for plot_idx, (metric, data_idx) in enumerate(zip(metrics, metric_indices)):
    ax = axes[plot_idx]
    
    # 提取该指标的所有模型数据
    metric_values = [processed_data[model][data_idx] for model in model_names]
    
    # 绘制纵向柱状图，每个柱子不同的纹理
    bars = []
    for i, (name, value) in enumerate(zip(model_names, metric_values)):
        bar = ax.bar(i, value, color=bar_colors[i], 
                    edgecolor='gray', linewidth=2,
                    hatch=hatches[i], alpha=0.75, width=0.7)
        bars.append(bar)
    
    # --- 绘制提升标注 ---
    if teacher_idx is not None and gpt5m_idx is not None:
        teacher_value = metric_values[teacher_idx]
        gpt5m_value = metric_values[gpt5m_idx]
        
        if teacher_value > gpt5m_value:
            # 计算提升百分比
            improvement = ((teacher_value - gpt5m_value) / gpt5m_value) * 100
            
            # x坐标
            x_start_line = x_pos[gpt5m_idx] 
            x_end_line = x_pos[teacher_idx] 
            
            # 顶部横线稍微上移一点
            v_offset = 0.001 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            
            # 绘制顶部横线
            ax.plot([x_start_line, x_end_line], 
                    [teacher_value + v_offset, teacher_value + v_offset],
                    color='red', linewidth=2.2, linestyle='-', zorder=10)
            
            # 绘制垂直虚线箭头
            arrow_x_position = x_pos[gpt5m_idx]
            
            ax.annotate('', 
                        xy=(arrow_x_position, teacher_value + v_offset),
                        xytext=(arrow_x_position, gpt5m_value),
                        arrowprops=dict(
                            arrowstyle='->',
                            color='red',
                            lw=2.2,
                            linestyle='--',
                            shrinkA=0, 
                            shrinkB=0
                        ),
                        zorder=10)
            
            # 添加百分比文本
            text_x = arrow_x_position - 0.15
            text_y = gpt5m_value + (teacher_value + v_offset - gpt5m_value) / 2
            
            ax.text(text_x, text_y, 
                    f'+{improvement:.1f}%', 
                    fontsize=16, color='red', weight='bold',
                    ha='right', va='center',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.3'))

    # 设置x轴标签（指标名称）
    ax.set_xlabel(metric, fontsize=11, weight='bold', labelpad=10)
    ax.set_xticks(x_pos)
    
    # 简化模型名称显示
    short_names = []
    for name in model_names:
        if 'Gemini' in name:
            short_names.append('Gemini')
        elif 'GPT' in name:
            short_names.append('GPT-5M')
        elif 'Qwen' in name:
            short_names.append('Qwen')
        elif 'w/o DPO' in name:
            short_names.append('w/o DPO')
        elif 'w/o SFT' in name:
            short_names.append('w/o SFT')
        elif 'Teacher' in name:
            short_names.append('Teacher')
        else:
            short_names.append(name)
    
    # x轴标签（模型名称）
    ax.set_xticklabels(short_names, fontsize=11, rotation=25, ha='center')
    
    # y轴标签
    ax.set_ylabel('Score', fontsize=13, weight='bold', labelpad=8)
    
    # 添加网格线
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.9, color='gray')
    ax.set_axisbelow(True)
    
    # 设置y轴范围
    max_val = max(metric_values)
    # 让第三个图（PAD）使用第一个图（CAR）的y轴范围
    if plot_idx == 2:  # 第三个图是PAD
        # 获取第一个图的y轴范围
        first_ax_ylim = axes[0].get_ylim()
        ax.set_ylim(first_ax_ylim)
    else:
        ax.set_ylim(0, max_val * 1.35)
    
    # 设置y轴刻度格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 移除外边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=11, width=0, length=5)

# 调整子图间距
plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)

# 保存图像
filename = 'ablation_study_comparison.png'
plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
print(f"\n✓ 消融实验柱状图已保存为 {filename}")

# 同时保存PDF版本
filename_pdf = 'ablation_study_comparison.pdf'
plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
print(f"✓ PDF版本已保存为 {filename_pdf}")

plt.show()
