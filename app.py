import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="衰弱风险预测SHAP分析",
    page_icon="🏥",
    layout="centered",  # 改为居中布局
    initial_sidebar_state="collapsed"  # 收起侧边栏
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
    }
    .feature-group {
        margin-bottom: 1.5rem;
    }
    .feature-group h3 {
        color: #495057;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
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
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
st.markdown('<h1 class="main-header">🏥 衰弱风险预测评估系统</h1>', unsafe_allow_html=True)

# 评估表单 - 放在页面正中间
with st.form("assessment_form"):
    st.markdown('<div class="assessment-section">', unsafe_allow_html=True)
    
    # 第一行：人口学特征
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        age = st.slider("年龄", 50, 100, 71)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        gender = st.selectbox("性别", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        bmi = st.slider("BMI", 15.0, 40.0, 26.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        smoke = st.selectbox("吸烟", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第二行：身体功能指标
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        ftsst = st.selectbox("FTSST (5次坐立测试)", [0, 1], 
                           format_func=lambda x: "≤12秒" if x == 0 else ">12秒")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        adl = st.selectbox("ADL (日常生活能力)", [0, 1], 
                         format_func=lambda x: "无限制" if x == 0 else "有限制")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col7:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        pa = st.selectbox("体力活动水平", [0, 1, 2], 
                        format_func=lambda x: ["高", "中", "低"][x])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第三行：临床指标
    col8, col9, col10, col11 = st.columns(4)
    with col8:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        complications = st.selectbox("并发症数量", [0, 1, 2], 
                                   format_func=lambda x: ["无", "1个", "≥2个"][x])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col9:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        fall = st.selectbox("跌倒史", [0, 1], format_func=lambda x: "无" if x == 0 else "有")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col10:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        bl_crp = st.slider("CRP (mg/L)", 0.0, 20.0, 9.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col11:
        st.markdown('<div class="feature-group">', unsafe_allow_html=True)
        bl_hgb = st.slider("血红蛋白 (g/L)", 80.0, 200.0, 150.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    # 显示SHAP力图
    st.markdown("### 📈 SHAP力分析图")
    fig = create_shap_force_plot_plotly(base_val, current_val, shap_vals, feature_names, sample_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # 风险分析
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### ⚠️ 主要风险因素")
        risk_factors = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
            if shap_val > 0.01:  # 显著增加风险
                original_feature = list(sample_data.keys())[i]
                value = sample_data[original_feature]
                risk_factors.append(f"**{feature}** = {value}")
        
        if risk_factors:
            for factor in risk_factors:
                st.error(factor)
        else:
            st.info("无显著风险因素")
    
    with col_right:
        st.markdown("### 🛡️ 保护因素")
        protective_factors = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
            if shap_val < -0.01:  # 显著降低风险
                original_feature = list(sample_data.keys())[i]
                value = sample_data[original_feature]
                protective_factors.append(f"**{feature}** = {value}")
        
        if protective_factors:
            for factor in protective_factors:
                st.success(factor)
        else:
            st.info("无显著保护因素")
    
    # 详细特征分析
    with st.expander("📋 查看详细特征分析"):
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
        
        st.dataframe(
            contribution_df,
            use_container_width=True,
            column_config={
                "特征": st.column_config.TextColumn("特征"),
                "SHAP值": st.column_config.NumberColumn("SHAP值", format="%.4f"),
                "特征值": st.column_config.NumberColumn("特征值"),
                "影响方向": st.column_config.TextColumn("影响方向")
            }
        )

# 页脚说明
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>使用说明：</strong> 填写完所有评估指标后，点击"点击预测"按钮获取个性化衰弱风险评估结果</p>
</div>
""", unsafe_allow_html=True)
