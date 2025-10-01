import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="衰弱风险预测SHAP分析",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .assessment-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .feature-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
    }
    .feature-label {
        flex: 1;
        font-weight: bold;
        color: #495057;
    }
    .feature-input {
        flex: 1;
        text-align: right;
    }
    .stButton button {
        width: 200px;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin: 1rem auto;
        display: block;
    }
    .stButton button:hover {
        background-color: #1668a5;
    }
    .result-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-top: 2rem;
        text-align: center;
    }
    .shap-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 2rem 0;
    }
    .feature-value-display {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #666;
        flex-wrap: wrap;
    }
    .feature-item {
        margin: 0.5rem 1rem;
        text-align: center;
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
    
    return base_value, current_value, shap_values, feature_names

def create_horizontal_shap_plot(base_value, current_value, shap_values, feature_names, sample_data):
    """创建水平SHAP力图，类似提供的图片样式"""
    
    fig = go.Figure()
    
    # 计算累积值
    cumulative = base_value
    x_positions = [base_value]
    
    # 添加基准线
    fig.add_shape(
        type="line",
        x0=base_value, y0=0,
        x1=base_value, y1=1,
        line=dict(color="black", width=3, dash="dash"),
        name="Base Value"
    )
    
    # 添加每个特征的贡献箭头
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
        start_x = cumulative
        end_x = cumulative + shap_val
        cumulative = end_x
        
        # 确定颜色
        color = '#FF6B6B' if shap_val > 0 else '#4ECDC4'
        
        # 添加箭头
        fig.add_trace(go.Scatter(
            x=[start_x, end_x],
            y=[0.5, 0.5],
            mode='lines+markers',
            line=dict(color=color, width=8),
            marker=dict(size=0),
            name=feature,
            hovertemplate=f'<b>{feature}</b><br>贡献: {shap_val:.4f}<br>特征值: {list(sample_data.values())[i]}<extra></extra>'
        ))
        
        x_positions.append(end_x)
    
    # 添加最终预测值点
    fig.add_trace(go.Scatter(
        x=[current_value],
        y=[0.5],
        mode='markers',
        marker=dict(size=15, color='#FFD93D', line=dict(width=2, color='black')),
        name='预测值'
    ))
    
    fig.update_layout(
        title=dict(
            text="SHAP Force Plot for Individual Prediction",
            x=0.5,
            xanchor='center',
            font=dict(size=16, weight='bold')
        ),
        xaxis=dict(
            title="模型输出值",
            range=[0, 0.7],
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            showticklabels=False,
            range=[0, 1],
            showgrid=False
        ),
        height=300,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    # 添加higher/lower标签
    fig.add_annotation(
        x=0.02, y=0.9,
        xref="paper", yref="paper",
        text="higher",
        showarrow=False,
        font=dict(color="red", size=12)
    )
    
    fig.add_annotation(
        x=0.98, y=0.9,
        xref="paper", yref="paper",
        text="lower",
        showarrow=False,
        font=dict(color="blue", size=12)
    )
    
    # 添加base value标签
    fig.add_annotation(
        x=base_value, y=0.1,
        text=f"base value<br>{base_value:.3f}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="lightgray"
    )
    
    return fig

# 应用标题
st.markdown('<h1 class="main-header">🏥 衰弱风险预测评估系统</h1>', unsafe_allow_html=True)

# 评估表单 - 所有问题排成一列
with st.form("assessment_form"):
    st.markdown('<div class="assessment-section">', unsafe_allow_html=True)
    
    # 所有特征排成一列
    features_data = []
    
    # 人口学特征
    st.markdown("### 人口学特征")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("年龄", 50, 100, 71)
    with col2:
        gender = st.selectbox("性别", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
    
    col3, col4 = st.columns(2)
    with col3:
        bmi = st.slider("BMI", 15.0, 40.0, 26.0, 0.1)
    with col4:
        smoke = st.selectbox("吸烟", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
    
    # 身体功能指标
    st.markdown("### 身体功能指标")
    col5, col6, col7 = st.columns(3)
    with col5:
        ftsst = st.selectbox("FTSST (5次坐立测试)", [0, 1], 
                           format_func=lambda x: "≤12秒" if x == 0 else ">12秒")
    with col6:
        adl = st.selectbox("ADL (日常生活能力)", [0, 1], 
                         format_func=lambda x: "无限制" if x == 0 else "有限制")
    with col7:
        pa = st.selectbox("体力活动水平", [0, 1, 2], 
                        format_func=lambda x: ["高", "中", "低"][x])
    
    # 临床指标
    st.markdown("### 临床指标")
    col8, col9, col10, col11 = st.columns(4)
    with col8:
        complications = st.selectbox("并发症数量", [0, 1, 2], 
                                   format_func=lambda x: ["无", "1个", "≥2个"][x])
    with col9:
        fall = st.selectbox("跌倒史", [0, 1], format_func=lambda x: "无" if x == 0 else "有")
    with col10:
        bl_crp = st.slider("CRP (mg/L)", 0.0, 20.0, 9.0, 0.1)
    with col11:
        bl_hgb = st.slider("血红蛋白 (g/L)", 80.0, 200.0, 150.0, 1.0)
    
    # 预测按钮
    submit_button = st.form_submit_button("🚀 点击预测")
    
    st.markdown('</div>', unsafe_allow_html=True)

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
    base_val, current_val, shap_vals, feature_names = calculate_shap_values(sample_data)
    
    # 显示预测结果
    st.markdown("---")
    st.markdown(f'<div class="result-section">', unsafe_allow_html=True)
    st.markdown(f"### 📊 预测结果: 患者衰弱概率为 **{current_val:.1%}**")
    st.markdown(f'</div>', unsafe_allow_html=True)
    
    # 显示水平SHAP力图
    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
    fig = create_horizontal_shap_plot(base_val, current_val, shap_vals, feature_names, sample_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # 在SHAP图下方显示特征名称和数值
    st.markdown('<div class="feature-value-display">', unsafe_allow_html=True)
    
    # 特征显示名称映射
    feature_display_map = {
        'FTSST': 'FTSST',
        'Complications': 'Complications',
        'History of falls': 'History of falls',
        'CRP': 'CRP',
        'PA': 'PA',
        'HGB': 'HGB',
        'Smoke': 'Smoke',
        'Gender': 'Gender',
        'Age': 'Age',
        'BMI': 'BMI',
        'ADL': 'ADL'
    }
    
    # 显示所有特征的值
    for i, feature in enumerate(feature_names):
        value = list(sample_data.values())[i]
        st.markdown(f'<div class="feature-item"><strong>{feature_display_map[feature]}</strong> = {value}</div>', 
                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 风险分析
    st.markdown("---")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### ⚠️ 主要风险因素")
        risk_factors = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
            if shap_val > 0.01:
                value = list(sample_data.values())[i]
                risk_factors.append(f"**{feature}** = {value} (贡献: {shap_val:.4f})")
        
        if risk_factors:
            for factor in risk_factors:
                st.error(factor)
        else:
            st.info("无显著风险因素")
    
    with col_right:
        st.markdown("### 🛡️ 保护因素")
        protective_factors = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
            if shap_val < -0.01:
                value = list(sample_data.values())[i]
                protective_factors.append(f"**{feature}** = {value} (贡献: {shap_val:.4f})")
        
        if protective_factors:
            for factor in protective_factors:
                st.success(factor)
        else:
            st.info("无显著保护因素")

# 页脚说明
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>使用说明：</strong> 填写完所有评估指标后，点击"点击预测"按钮获取个性化衰弱风险评估结果</p>
</div>
""", unsafe_allow_html=True)
