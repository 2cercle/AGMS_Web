# -*- coding: utf-8 -*-
"""
AGMS Sensor Analysis Dashboard (Auto Lag Optimization + Full Clarke Grid)
Powered by Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io
import platform

# -----------------------------------------------------------------------------
# [1] 페이지 및 폰트 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AGMS 센서 분석기",
    page_icon="🩸",
    layout="wide"
)

# 한글 폰트 설정
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# [2] Helper 함수: Clarke Error Grid Logic
# -----------------------------------------------------------------------------
def get_clarke_zone(ref, pred):
    if ref == 0: return 'B'
    abs_diff = abs(ref - pred)
    
    # Zone A
    if abs_diff <= 0.2 * ref: return 'A'
    if ref < 70 and abs_diff <= 15: return 'A'

    # Zone E
    if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70): return 'E'
    
    # Zone D
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 70 and 70 <= pred <= 180): return 'D'
    
    # Zone C
    if (pred > ref + 110) or (pred < ref - 110): return 'C'

    # Zone B
    return 'B'

def plot_clarke_grid(y_test, y_pred, ax):
    zones = [get_clarke_zone(r, p) for r, p in zip(y_test, y_pred)]
    zone_counts = {z: zones.count(z) for z in ['A', 'B', 'C', 'D', 'E']}
    total = len(zones)
    
    colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#d62728', 'E': '#9467bd'}
    
    for z in ['A', 'B', 'C', 'D', 'E']:
        mask = [zone == z for zone in zones]
        if sum(mask) > 0:
            ax.scatter(
                y_test[mask], y_pred[mask], 
                c=colors[z], s=25, alpha=0.6, edgecolors='white', linewidth=0.5,
                label=f'Zone {z}: {zone_counts[z]} ({zone_counts[z]/total*100:.1f}%)'
            )

    ax.set_title("Clarke Error Grid Analysis", fontsize=12, fontweight='bold')
    ax.set_xlabel("Reference Glucose (mg/dL)")
    ax.set_ylabel("Sensor Glucose (mg/dL)")
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_aspect('equal')

    # Grid Lines
    ax.plot([0, 400], [0, 400], 'k--', lw=1.5, alpha=0.7)
    ax.plot([0, 333.3], [0, 400], 'k-', lw=1) # y=1.2x
    ax.plot([0, 400], [0, 320], 'k-', lw=1)   # y=0.8x
    ax.plot([0, 400], [180, 180], 'k-', lw=1)
    ax.plot([0, 400], [70, 70], 'k-', lw=1)
    ax.plot([180, 180], [0, 400], 'k-', lw=1)
    ax.plot([70, 70], [0, 400], 'k-', lw=1)
    ax.plot([240, 240], [70, 180], 'k-', lw=1)

    # Zone Labels
    ax.text(30, 10, 'E', fontsize=12, color='red', fontweight='bold')
    ax.text(350, 350, 'A', fontsize=12, color='green', fontweight='bold')
    ax.text(280, 200, 'B', fontsize=10, color='blue')
    ax.text(350, 120, 'D', fontsize=10, color='red')
    ax.text(30, 350, 'E', fontsize=12, color='red', fontweight='bold')
    ax.text(130, 350, 'C', fontsize=10, color='orange')

    ax.legend(loc='upper left', fontsize='small', frameon=True)
    ax.grid(False)

# -----------------------------------------------------------------------------
# [3] 데이터 처리 로직 (분리 및 최적화)
# -----------------------------------------------------------------------------

@st.cache_data
def load_and_clean_data(libre_file, sensor_files, warmup_hours):
    """
    1단계: 파일 로드 및 기본 정제 (Lag 적용 전 단계)
    """
    # --- 1. 리브레 로드 ---
    try:
        if libre_file.name.endswith('.xlsx'):
            libre_df = pd.read_excel(libre_file)
        else:
            libre_df = pd.read_csv(libre_file, skiprows=1)
            
        col_map = {
            'Device Timestamp': 'ts', 'Historic Glucose mg/dL': 'gl', 
            'Scan Glucose mg/dL': 'gl_scan', 'Timestamp': 'ts', 'Glucose': 'gl'
        }
        libre_df = libre_df.rename(columns=lambda x: col_map.get(x, x))
        libre_df['ts'] = pd.to_datetime(libre_df['ts'], errors='coerce')
        libre_df = libre_df.dropna(subset=['ts'])
        
        if 'gl' not in libre_df.columns and 'gl_scan' in libre_df.columns:
            libre_df['gl'] = libre_df['gl_scan']
            
        libre_df['gl'] = pd.to_numeric(libre_df['gl'], errors='coerce').interpolate()
        libre_df = libre_df.sort_values('ts')
    except Exception as e:
        return None, None, f"리브레 파일 오류: {str(e)}"

    # --- 2. 센서 로드 및 칼만 필터 ---
    sensor_list = []
    use_cols = ['experiment_date', 'value_current', 'value_ae', 'value_temperature']
    
    for sf in sensor_files:
        try:
            sf.seek(0) # 파일 포인터 초기화 (중요)
            temp = pd.read_csv(sf, usecols=lambda c: c in use_cols)
            sensor_list.append(temp)
        except: pass

    if not sensor_list:
        return None, None, "유효한 센서 데이터가 없습니다."

    sensor_df = pd.concat(sensor_list, ignore_index=True)
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['experiment_date'], errors='coerce')
    sensor_df = sensor_df.dropna(subset=['timestamp']).sort_values('timestamp')

    cols = ['value_current', 'value_ae', 'value_temperature']
    sensor_df[cols] = sensor_df[cols].ffill().bfill()
    
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    for c in cols:
        sensor_df[f'{c}_kf'], _ = kf.smooth(sensor_df[c].values)

    start_t = sensor_df['timestamp'].min()
    sensor_df['hours_since_start'] = (sensor_df['timestamp'] - start_t).dt.total_seconds() / 3600.0
    sensor_df = sensor_df[sensor_df['hours_since_start'] > warmup_hours]
    
    if sensor_df.empty:
        return None, None, f"Warm-up({warmup_hours}h) 이후 데이터 없음"

    return libre_df, sensor_df, None

def merge_with_lag(libre_df, sensor_df, lag_minutes):
    """
    2단계: 특정 Lag를 적용하여 병합 (반복 호출용)
    """
    temp_libre = libre_df.copy()
    # 리브레 시간을 뒤로 당김 = 센서가 리브레보다 늦게 반응함을 보정
    temp_libre['ts_merge'] = temp_libre['ts'] - pd.Timedelta(minutes=lag_minutes)
    temp_libre = temp_libre.sort_values('ts_merge')
    
    merged = pd.merge_asof(temp_libre, sensor_df, left_on='ts_merge', right_on='timestamp',
                           direction='nearest', tolerance=pd.Timedelta('15min'))
    
    return merged.dropna(subset=['gl', 'value_current_kf'])

def train_and_evaluate(df):
    """
    3단계: 모델 학습 및 정확도(15/15%) 반환
    """
    if df.empty or len(df) < 10: return 0, 0, None, None, None

    features = ['value_current_kf', 'value_ae_kf', 'value_temperature_kf', 'hours_since_start']
    X = df[features]
    y = df['gl']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) # 속도를 위해 estimators 조절
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    def check_15_15(yt, yp):
        if yt < 100: return abs(yt - yp) <= 15
        else: return abs(yt - yp) / yt <= 0.15
    
    acc_15 = (sum([check_15_15(yt, yp) for yt, yp in zip(y_test, y_pred)]) / len(y_test)) * 100
    
    return acc_15, model, X_test, y_test, y_pred

# -----------------------------------------------------------------------------
# [4] 사이드바 UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 1. 데이터 입력")
    uploaded_libre = st.file_uploader("1) 리브레 데이터 (엑셀/CSV)", type=['csv', 'xlsx'])
    uploaded_sensors = st.file_uploader("2) 센서 데이터 (CSV, 다중 선택)", type=['csv'], accept_multiple_files=True)
    
    st.header("⚙️ 2. 분석 설정")
    
    # 최적화 옵션 추가
    use_auto_lag = st.checkbox("✅ 최적 시간 지연 자동 탐색", value=False, help="5~15분 범위에서 정확도가 가장 높은 시간을 자동으로 찾습니다.")
    
    if use_auto_lag:
        st.info("⏱️ 5분 ~ 15분 범위를 탐색합니다.")
        lag_min = 0 # Placeholder
    else:
        lag_min = st.number_input("시간 지연 (분)", value=15, step=1)
        
    warmup_hr = st.number_input("초기 제거 (시간)", value=24, step=1)
    
    st.header("📝 3. 리포트 정보")
    memo = st.text_input("실험 메모", placeholder="실험 내용 입력")
    
    st.divider()
    run_btn = st.button("분석 실행 🚀", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# [5] 메인 대시보드
# -----------------------------------------------------------------------------
if run_btn:
    if uploaded_libre and uploaded_sensors:
        report_title = f"📊 AGMS 분석 결과: {memo}" if memo else "📊 AGMS 분석 결과"
        st.title(report_title)
        
        # 1. 데이터 로드 (공통)
        with st.spinner('데이터 로딩 및 전처리 중...'):
            libre_df, sensor_df, err = load_and_clean_data(uploaded_libre, uploaded_sensors, warmup_hr)
            
        if err:
            st.error(err)
        else:
            final_lag = lag_min
            final_df = None
            final_results = None # (acc, model, X_test, y_test, y_pred)
            
            # 2. 자동 최적화 로직
            if use_auto_lag:
                best_acc = -1
                best_lag = 5
                
                progress_text = "최적 시간 지연(Lag) 탐색 중... (5~15분)"
                my_bar = st.progress(0, text=progress_text)
                
                # 탐색 범위: 5분 ~ 15분
                search_range = range(5, 16)
                total_steps = len(search_range)
                
                for i, temp_lag in enumerate(search_range):
                    # Merge
                    temp_df = merge_with_lag(libre_df, sensor_df, temp_lag)
                    # Train & Eval
                    acc, model, xt, yt, yp = train_and_evaluate(temp_df)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_lag = temp_lag
                        final_df = temp_df
                        final_results = (acc, model, xt, yt, yp)
                    
                    my_bar.progress((i + 1) / total_steps, text=f"Lag {temp_lag}분 테스트 중... (현재 최고 정확도: {best_acc:.2f}%)")
                
                my_bar.empty()
                st.success(f"🎯 최적 지연 시간 발견: **{best_lag}분** (15/15% 정확도: {best_acc:.2f}%)")
                final_lag = best_lag
                
            else:
                # 수동 모드
                with st.spinner('분석 수행 중...'):
                    final_df = merge_with_lag(libre_df, sensor_df, lag_min)
                    acc, model, xt, yt, yp = train_and_evaluate(final_df)
                    final_results = (acc, model, xt, yt, yp)
                    
            # 3. 결과 시각화 (공통)
            if final_results and final_results[3] is not None:
                acc_15, model, X_test, y_test, y_pred = final_results
                
                # R2, MARD 재계산 (최종 모델 기준)
                r2 = r2_score(y_test, y_pred)
                mard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # KPI 표시
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("MARD (오차율)", f"{mard:.2f}%", delta_color="inverse")
                kpi2.metric("15/15% 정확도", f"{acc_15:.2f}%")
                kpi3.metric("R-Squared", f"{r2:.4f}")
                kpi4.metric(f"적용된 지연 시간", f"{final_lag}분")
                
                st.divider()

                # --- Graph 1: Plotly Time Series ---
                st.subheader(f"📈 혈당 그래프 (Lag {final_lag}분 적용)")
                ref_values = y_test.values
                upper_bound = [r + 15 if r < 100 else r * 1.15 for r in ref_values]
                lower_bound = [r - 15 if r < 100 else r * 0.85 for r in ref_values]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test.index, y=lower_bound, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=y_test.index, y=upper_bound, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='허용 오차 범위', hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='실제 혈당 (Reference)', line=dict(color='black', width=2)))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='AI 예측 (Predicted)', line=dict(color='#d62728', width=2, dash='dot')))
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center'))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Graph 2: Clarke & Hist ---
                c1, c2 = st.columns(2)
                with c1:
                    fig_clarke, ax = plt.subplots(figsize=(6, 6))
                    plot_clarke_grid(y_test.values, y_pred, ax)
                    st.pyplot(fig_clarke)
                with c2:
                    st.markdown("##### 📊 오차 분포 (Residuals)")
                    errors = y_pred - y_test
                    fig_hist, ax2 = plt.subplots(figsize=(6, 6))
                    sns.histplot(errors, kde=True, bins=25, color='orange', ax=ax2)
                    ax2.axvline(0, color='black', linestyle='--')
                    ax2.set_xlabel('Error (Predicted - Reference)')
                    st.pyplot(fig_hist)
                
                # --- Excel Download ---
                st.subheader("📥 리포트 다운로드")
                res_df = final_df.copy()
                res_df['Predicted_Glucose'] = np.nan
                res_df.loc[y_test.index, 'Predicted_Glucose'] = y_pred
                res_df['Error_Diff'] = res_df['Predicted_Glucose'] - res_df['gl']
                res_df['Error_Pct'] = (res_df['Error_Diff'] / res_df['gl']) * 100
                zones = [get_clarke_zone(r, p) if pd.notnull(p) else np.nan for r, p in zip(res_df['gl'], res_df['Predicted_Glucose'])]
                res_df['Clarke_Zone'] = zones
                
                save_cols = ['ts', 'gl', 'Predicted_Glucose', 'Clarke_Zone', 'Error_Diff', 'Error_Pct'] + [c for c in res_df.columns if c not in ['ts', 'gl', 'Predicted_Glucose', 'Clarke_Zone', 'Error_Diff', 'Error_Pct']]
                res_df = res_df[save_cols]

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Raw_Data')
                    summary = pd.DataFrame({
                        'Item': ['Memo', 'Applied Lag (min)', 'R2', 'MARD', '15/15 Accuracy'],
                        'Value': [memo, final_lag, r2, f"{mard:.2f}%", f"{acc_15:.2f}%"]
                    })
                    summary.to_excel(writer, index=False, sheet_name='Summary')
                    
                st.download_button(label="📊 결과 엑셀 다운로드", data=buffer.getvalue(), file_name=f"AGMS_Result_{memo}.xlsx" if memo else "AGMS_Result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            else:
                st.error("분석할 데이터가 충분하지 않습니다.")
    else:
        st.warning("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")
else:
    st.info("👈 파일 업로드 후 '분석 실행' 버튼을 눌러주세요.")
