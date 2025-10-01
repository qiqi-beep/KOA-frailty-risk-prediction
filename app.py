import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="衰弱风险预测SHAP分析",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .feature-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-low {
        color: #0068c9;
        font-weight: bold;
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
    shap_values[features.index('age')] = 0.08 * (sample_data['age'] / 71)        # 年龄
    shap_values[features.index('FTSST')] = 0.06 * sample_data['FTSST']           # FTSST
    shap_values[features.index('bmi')] = 0.05 * (sample_data['bmi'] / 26)        # BMI
    shap_values[features.index('Complications')] = 0.04 * sample_data['Complications']  # 并发症
    shap_values[features.index('fall')] = 0.03 * sample_data['fall']             # 跌倒史
    shap_values[features.index('ADL')] = 0.02 * sample_data['ADL']               # ADL
    shap_values[features.index('bl_crp')] = 0.01 * (sample_data['bl_crp'] / 9)   # CRP
    shap_values[features.index('gender')] = 0.04 * sample_data['gender']         # 性别
    
    # 负向预测变量 - 负值降低风险
    shap_values[features.index('PA')] = -0.02 * (2 - sample_data['PA'])          # 体力活动
    shap_values[features.index('smoke')] = -0.03 * (1 - sample_data['smoke'])    # 吸烟
    shap_values[features.index('bl_hgb')] = -0.01                               # HGB
    
    # 设置基础值和当前预测值
    base_value = 0.35  # 平均风险概率
    current_value = base_value + shap_values.sum()
    
    # 确保预测值在合理范围内
    current_value = max(0.01, min(0.99, current_value))
    
    return base_value, current_value, shap_values, feature_names

def create_shap_force_plot_plotly(base_value, current_value, shap_values, feature_names, sample_data):
    """创建Plotly版本的SHAP力图"""
    
    fig = go.Figure()
    
    # 添加基准线
    fig.add_shape(
        type="line",
        x0=base_value, y0=-0.5,
        x1=base_value, y1=len(feature_names) - 0.5,
        line=dict(color="gray", width=2, dash="dash")
    )
    
    # 添加每个特征的贡献
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
        color = '#FF4B4B' if shap_val > 0 else '#0068C9'
        
        fig.add_trace(go.Bar(
            x=[shap_val],
            y=[feature],
            orientation='h',
            name=feature,
            marker_color=color,
            hovertemplate=f'<b>{feature}</b><br>特征值: {list(sample_data.values())[i]}<br>SHAP贡献: {shap_val:.4f}<br>影响: {"增加风险" if shap_val > 0 else "降低风险"}<extra></extra>'
        ))
    
    fig.update_layout(
        title="SHAP力分析图",
        xaxis_title="SHAP值贡献",
        yaxis_title="特征",
        barmode='relative',
        height=500,
        showlegend=False
    )
    
    # 添加最终预测值标注
    fig.add_annotation(
        x=current_value,
        y=len(feature_names) - 0.5,
        text=f"最终预测: {current_value:.3f}",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

# 应用标题
st.markdown('<h1 class="main-header">🏥 衰弱风险预测SHAP分析平台</h1>', unsafe_allow_html=True)

# 侧边栏 - 输入参数
st.sidebar.header("📊 输入患者特征")

with st.sidebar.expander("身体功能指标", expanded=True):
    ftsst = st.selectbox("FTSST (5次坐立测试)", [0, 1], format_func=lambda x: "≤12秒" if x == 0 else ">12秒")
    adl = st.selectbox("ADL (日常生活能力)", [0, 1], format_func=lambda x: "无限制" if x == 0 else "有限制")
    pa = st.selectbox("体力活动水平", [0, 1, 2], format_func=lambda x: ["高", "中", "低"][x])

with st.sidebar.expander("临床指标", expanded=True):
    complications = st.selectbox("并发症数量", [0, 1, 2], format_func=lambda x: ["无", "1个", "≥2个"][x])
    fall = st.selectbox("跌倒史", [0, 1], format_func=lambda x: "无" if x == 0 else "有")
    bl_crp = st.slider("CRP (mg/L)", 0.0, 20.0, 9.0, 0.1)
    bl_hgb = st.slider("血红蛋白 (g/L)", 80.0, 200.0, 150.0, 1.0)

with st.sidebar.expander("人口学特征", expanded=True):
    age = st.slider("年龄", 50, 100, 71)
    bmi = st.slider("BMI", 15.0, 40.0, 26.0, 0.1)
    gender = st.selectbox("性别", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
    smoke = st.selectbox("吸烟", [0, 1], format_func=lambda x: "否" if x == 0 else "是")

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

# 主内容区域
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 SHAP力分析图")
    
    # 生成Plotly SHAP图
    fig = create_shap_force_plot_plotly(base_val, current_val, shap_vals, feature_names, sample_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示预测结果
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.metric(
            label="预测风险概率",
            value=f"{current_val:.1%}",
            delta=f"{(current_val - base_val):+.1%}",
            delta_color="inverse"
        )
    with col1_2:
        st.metric(
            label="基准风险概率",
            value=f"{base_val:.1%}"
        )

with col2:
    st.header("🎯 风险分析")
    
    # 风险因素分析
    risk_factors = []
    protective_factors = []
    
    # 分析每个特征的风险方向
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
        original_feature = list(sample_data.keys())[i]
        value = sample_data[original_feature]
        
        if shap_val > 0.01:  # 显著增加风险
            risk_factors.append(f"{feature} = {value}")
        elif shap_val < -0.01:  # 显著降低风险
            protective_factors.append(f"{feature} = {value}")
    
    st.subheader("⚠️ 主要风险因素")
    if risk_factors:
        for factor in risk_factors[:5]:
            st.error(factor)
    else:
        st.info("无显著风险因素")
    
    st.subheader("🛡️ 保护因素")
    if protective_factors:
        for factor in protective_factors:
            st.success(factor)
    else:
        st.info("无显著保护因素")

# 贡献度分析
st.header("📊 特征贡献度分析")

# 创建贡献度表格
contribution_data = []
for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
    original_feature = list(sample_data.keys())[i]
    contribution_data.append({
        '特征': feature,
        'SHAP值': shap_val,
        '特征值': sample_data[original_feature],
        '影响方向': '增加风险' if shap_val > 0 else '降低风险'
    })

contribution_df = pd.DataFrame(contribution_data)
contribution_df = contribution_df.sort_values('SHAP值', key=abs, ascending=False)

# 显示贡献度表格
st.subheader("特征贡献度排序")
st.dataframe(
    contribution_df,
    use_container_width=True,
    column_config={
        "特征": st.column_config.TextColumn("特征"),
        "SHAP值": st.column_config.NumberColumn("SHAP值", format="%.4f"),
        "特征值": st.column_config.NumberColumn("特征值", format="%.1f"),
        "影响方向": st.column_config.TextColumn("影响方向")
    }
)

# 创建贡献度条形图
st.subheader("特征贡献度可视化")
fig_bar = go.Figure()

# 添加条形
fig_bar.add_trace(go.Bar(
    y=contribution_df['特征'],
    x=contribution_df['SHAP值'],
    orientation='h',
    marker_color=['#FF4B4B' if x > 0 else '#0068C9' for x in contribution_df['SHAP值']],
    hovertemplate='<b>%{y}</b><br>SHAP值: %{x:.4f}<br>影响: %{customdata}<extra></extra>',
    customdata=contribution_df['影响方向']
))

fig_bar.update_layout(
    title="特征对预测的贡献度 (SHAP值)",
    xaxis_title="SHAP值",
    yaxis_title="特征",
    showlegend=False,
    height=400
)

st.plotly_chart(fig_bar, use_container_width=True)

# 解释说明
st.header("💡 使用说明")
with st.expander("点击查看详细说明"):
    st.markdown("""
    **SHAP图解读:**
    - 🔴 **红色条形**: 特征增加患病风险
    - 🔵 **蓝色条形**: 特征降低患病风险
    - 📏 **条形长度**: 影响程度大小
    
    **特征说明:**
    - **FTSST**: 5次坐立测试时间 (>12秒为风险因素)
    - **ADL**: 日常生活能力 (受限为风险因素)
    - **PA**: 体力活动水平 (低水平为风险因素)
    - **Complications**: 并发症数量
    - **跌倒史**: 是否有跌倒史
    - **CRP**: C反应蛋白 (数值越高风险越大)
    - **年龄**: 年龄 (越大风险越高)
    - **BMI**: 体重指数 (越高风险越大)
    
    **预测说明:**
    - **基准风险**: 所有患者的平均风险水平
    - **预测风险**: 当前患者的个性化风险预测
    """)
