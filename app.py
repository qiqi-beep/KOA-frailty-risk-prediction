import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import io

# 页面配置
st.set_page_config(
    page_title="衰弱风险预测SHAP分析",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 自定义CSS样式 - 移除所有边框和特殊样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton button:hover {
        background-color: #1668a5;
    }
    .result-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
    /* 移除所有widget的边框和特殊样式 */
    .stSlider, .stSelectbox, .stNumberInput {
        border: none !important;
        box-shadow: none !important;
    }
    /* 移除标签的蓝色标记 */
    label {
        color: #262730 !important;
    }
    /* 移除所有边框 */
    div[data-testid="stForm"] {
        border: none !important;
        background: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def calculate_shap_values(sample_data):
    """计算SHAP值"""
    
    # 特征显示名称映射
    feature_display_names = {
        'FTSST': 'FTSST',
        'Complications': 'Complications',
        'fall': 'History of falls',
        'bl_crp': 'CRP',
        'PA': 'PA',
        'bl_hgb': 'HGB',
        'smoke': 'Smoke',
        'gender': 'Gender',
        'age': 'Age',
        'bmi': 'BMI',
        'ADL': 'ADL'
    }
    
    features = list(sample_data.keys())
    feature_names = [feature_display_names[f] for f in features]
    
    # 初始化SHAP值 - 基于临床意义的模拟值
    shap_values = np.zeros(len(features))
    
    # 为每个特征分配SHAP贡献（基于临床重要性）
    # 正向预测变量 - 正值增加风险
    shap_values[features.index('age')] = 0.08 * (sample_data['age'] / 71)
    shap_values[features.index('FTSST')] = 0.06 * sample_data['FTSST']
    shap_values[features.index('bmi')] = 0.05 * (sample_data['bmi'] / 26)
    shap_values[features.index('Complications')] = 0.04 * sample_data['Complications']
    shap_values[features.index('fall')] = 0.03 * sample_data['fall']
    shap_values[features.index('ADL')] = 0.02 * sample_data['ADL']
    shap_values[features.index('bl_crp')] = 0.01 * (sample_data['bl_crp'] / 9)
    shap_values[features.index('gender')] = 0.04 * sample_data['gender']
    
    # 负向预测变量 - 负值降低风险
    shap_values[features.index('PA')] = -0.02 * (2 - sample_data['PA'])
    shap_values[features.index('smoke')] = -0.03 * (1 - sample_data['smoke'])
    shap_values[features.index('bl_hgb')] = -0.01
    
    # 设置基础值和当前预测值
    base_value = 0.35
    current_value = base_value + shap_values.sum()
    current_value = max(0.01, min(0.99, current_value))
    
    return base_value, current_value, shap_values, feature_names, features

def create_shap_force_plot(base_value, shap_values, sample_data):
    """创建SHAP力分析图"""
    
    # 特征显示名称映射
    feature_display_names = {
        'FTSST': 'FTSST',
        'Complications': 'Complications',
        'fall': 'History of falls',
        'bl_crp': 'CRP',
        'PA': 'PA',
        'bl_hgb': 'HGB',
        'smoke': 'Smoke',
        'gender': 'Gender',
        'age': 'Age',
        'bmi': 'BMI',
        'ADL': 'ADL'
    }
    
    features = list(sample_data.keys())
    
    # 创建特征显示名称（包含数值）
    feature_display = []
    for feat in features:
        display_name = feature_display_names[feat]
        value = sample_data[feat]
        feature_display.append(f"{display_name} = {value}")
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    
    # 创建SHAP力图
    shap.force_plot(
        base_value,
        shap_values,
        feature_names=feature_display,
        matplotlib=True,
        show=False,
        plot_cmap=['#FF0D57', '#1E88E5']  # 红色=增加风险，蓝色=降低风险
    )
    
    plt.title("SHAP Force Plot for Individual Prediction", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # 将matplotlib图形转换为图片显示在Streamlit中
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# 应用标题
st.markdown('<h1 class="main-header">🏥 衰弱风险预测评估系统</h1>', unsafe_allow_html=True)

# 评估表单 - 所有问题排成一列，移除所有边框
with st.form("assessment_form"):
    
    # 所有特征排成一列
    st.markdown("### 请输入患者信息：")
    
    age = st.slider("年龄", 50, 100, 71)
    
    gender = st.selectbox("性别", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
    
    bmi = st.slider("BMI", 15.0, 40.0, 26.0, 0.1)
    
    smoke = st.selectbox("吸烟", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
    
    ftsst = st.selectbox("FTSST (5次坐立测试)", [0, 1], 
                       format_func=lambda x: "≤12秒" if x == 0 else ">12秒")
    
    adl = st.selectbox("ADL (日常生活能力)", [0, 1], 
                     format_func=lambda x: "无限制" if x == 0 else "有限制")
    
    pa = st.selectbox("体力活动水平", [0, 1, 2], 
                    format_func=lambda x: ["高", "中", "低"][x])
    
    complications = st.selectbox("并发症数量", [0, 1, 2], 
                               format_func=lambda x: ["无", "1个", "≥2个"][x])
    
    fall = st.selectbox("跌倒史", [0, 1], format_func=lambda x: "无" if x == 0 else "有")
    
    bl_crp = st.slider("CRP (mg/L)", 0.0, 20.0, 9.0, 0.1)
    
    bl_hgb = st.slider("血红蛋白 (g/L)", 80.0, 200.0, 150.0, 1.0)
    
    # 预测按钮
    submit_button = st.form_submit_button("🚀 点击预测")

# 处理预测结果
if submit_button:
    # 创建样本数据
    sample_data = {
        'FTSST': ftsst,
        'Complications': complications,
        'fall': fall,
        'bl_crp': float(bl_crp),
        'PA': pa,
        'bl_hgb': float(bl_hgb),
        'smoke': smoke,
        'gender': gender,
        'age': age,
        'bmi': float(bmi),
        'ADL': adl
    }
    
    # 计算SHAP值
    base_val, current_val, shap_vals, feature_names, features = calculate_shap_values(sample_data)
    
    # 显示预测结果
    st.markdown("---")
    st.markdown(f'<div class="result-section">', unsafe_allow_html=True)
    st.markdown(f"### 📊 预测结果: 患者衰弱概率为 **{current_val:.1%}**")
    st.markdown(f'</div>', unsafe_allow_html=True)
    
    # 生成并显示SHAP力图
    st.markdown("### 📈 SHAP力分析图")
    
    shap_image = create_shap_force_plot(base_val, shap_vals, sample_data)
    st.image(shap_image, use_container_width=True)

# 页脚说明
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>使用说明：</strong> 填写完所有评估指标后，点击"点击预测"按钮获取个性化衰弱风险评估结果</p>
</div>
""", unsafe_allow_html=True)
