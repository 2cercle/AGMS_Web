# -*- coding: utf-8 -*-
"""
AGMS Sensor Analysis Dashboard (Manual Mode)
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

# 한글 폰트 설정 (OS별 자동 대응)
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# [2] 데이터 처리 로직
# -----------------------------------------------------------------------------
@st.cache_data
def process_data(libre_file, sensor_files, lag_minutes, warmup_hours):
    """
    파일명과 상관없이 첫 번째 인자는 리브레, 두 번째 인자는 센서 데이터로 처리
    """
    
    # --- 1. 리브레(Reference) 데이터 로드 ---
    try:
        # 엑셀/CSV 구분 로드
        if libre_file.name.endswith('.xlsx'):
            libre_df = pd.read_excel(libre_file)
        else:
            # 보통 기기 데이터는 헤더가 2번째 줄에 있으므로 skiprows=1 유지
            # (만약 헤더가 첫 줄이라면 skiprows=0으로 수정 필요)
            libre_df = pd.read_csv(libre_file, skiprows=1) 
            
        # 컬럼 매핑 (이름이 조금 달라도 처리되도록 유연성 확보)
        col_map = {
            'Device Timestamp': 'ts', 
            'Historic Glucose mg/dL': 'gl', 
            'Scan Glucose mg/dL': 'gl_scan',
            'Timestamp': 'ts', 
            'Glucose': 'gl'
        }
        libre_df = libre_df.rename(columns=lambda x: col_map.get(x, x))
        
        # 필수 전처리
        libre_df['ts'] = pd.to_datetime(libre_df['ts'], errors='coerce')
        libre_df = libre_df.dropna(subset=['ts'])
        
        # 스캔 혈당 병합 (Historic이 없으면 Scan 사용)
        if 'gl' not in libre_df.columns and 'gl_scan' in libre_df.columns:
            libre_df['gl'] = libre_df['gl_scan']
        
        # 숫자 변환 및 보간
        libre_df['gl'] = pd.to_numeric(libre_df['gl'], errors='coerce').interpolate()
        libre_df = libre_df.sort_values('ts')
        
        # ★ 시간 지연(Lag) 즉시 적용 (Manual)
        # 리브레 시간에서 지연 시간을 뺌으로써, 센서 데이터와 매칭 시점 조절
        libre_df['ts_merge'] = libre_df['ts'] - pd.Timedelta(minutes=lag_minutes)
        libre_df = libre_df.sort_values('ts_merge')
        
    except Exception as e:
        return None, None, f"리브레(정답지) 파일 처리 중 오류: {str(e)}"

    # --- 2. 센서(Raw) 데이터 로드 ---
    sensor_list = []
    # 센서 데이터에서 꼭 필요한 컬럼명
    use_cols = ['experiment_date', 'value_current', 'value_ae', 'value_temperature']
    
    for sf in sensor_files:
        try:
            # 필요한 컬럼만 쏙 뽑아서 읽기 (속도 최적화)
            temp = pd.read_csv(sf, usecols=lambda c: c in use_cols)
            sensor_list.append(temp)
        except:
            # 컬럼이 없는 엉뚱한 파일은 무시
            pass

    if not sensor_list:
        return None, None, "유효한 센서 데이터가 없습니다. (CSV 내 컬럼명 확인: experiment_date, value_current 등)"

    sensor_df = pd.concat(sensor_list, ignore_index=True)
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['experiment_date'], errors='coerce')
    sensor_df = sensor_df.dropna(subset=['timestamp']).sort_values('timestamp')

    # 전처리: 결측치 채우기 & 칼만 필터
    cols = ['value_current', 'value_ae', 'value_temperature']
    sensor_df[cols] = sensor_df[cols].ffill().bfill()
    
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    for c in cols:
        sensor_df[f'{c}_kf'], _ = kf.smooth(sensor_df[c].values)

    # 초기 안정화 시간(Warm-up) 제거
    start_t = sensor_df['timestamp'].min()
    sensor_df['hours_since_start'] = (sensor_df['timestamp'] - start_t).dt.total_seconds() / 3600.0
    sensor_df = sensor_df[sensor_df['hours_since_start'] > warmup_hours]
    
    if sensor_df.empty:
        return None, None, f"초기 {warmup_hours}시간 제거 후 남은 데이터가 없습니다."

    # --- 3. 데이터 병합 (Merge) ---
    merged = pd.merge_asof(libre_df, sensor_df, left_on='ts_merge', right_on='timestamp',
                           direction='nearest', tolerance=pd.Timedelta('15min'))
    
    final_df = merged.dropna(subset=['gl', 'value_current_kf'])
    
    if final_df.empty:
        return None, None, "데이터 매칭 실패. 정답지와 센서 데이터의 시간대가 겹치는지 확인하세요."
        
    return final_df, sensor_df, None

# -----------------------------------------------------------------------------
# [3] 사이드바 UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 1. 데이터 입력")
    
    uploaded_libre = st.file_uploader("1) 리브레 데이터 (엑셀/CSV)", type=['csv', 'xlsx'])
    uploaded_sensors = st.file_uploader("2) 센서 데이터 (CSV, 다중 선택)", type=['csv'], accept_multiple_files=True)
    
    st.header("⚙️ 2. 분석 설정")
    lag_min = st.number_input("시간 지연 (분)", value=15, step=1, help="센서가 혈액보다 얼마나 늦게 반응하는지 설정")
    warmup_hr = st.number_input("초기 제거 (시간)", value=24, step=1, help="부착 초기 불안정 구간 제외")
    
    st.header("📝 3. 리포트 정보")
    memo = st.text_input("실험 메모", placeholder="실험 내용 입력")
    
    st.divider()
    run_btn = st.button("분석 실행 🚀", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# [4] 메인 대시보드
# -----------------------------------------------------------------------------
if run_btn:
    if uploaded_libre and uploaded_sensors:
        # 타이틀 설정
        report_title = f"📊 AGMS 분석 결과: {memo}" if memo else "📊 AGMS 분석 결과"
        st.title(report_title)
        
        with st.spinner('데이터 병합 및 AI 분석 중...'):
            df, _, err = process_data(uploaded_libre, uploaded_sensors, lag_min, warmup_hr)
            
            if err:
                st.error(err)
            else:
                # --- 머신러닝 모델링 (Random Forest) ---
                features = ['value_current_kf', 'value_ae_kf', 'value_temperature_kf', 'hours_since_start']
                X = df[features]
                y = df['gl']
                
                # 시계열 순서 유지 분할 (Shuffle=False) -> 과적합 방지
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # --- 지표 계산 ---
                r2 = r2_score(y_test, y_pred)
                mard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # 15/15% 정확도
                def check_15_15(yt, yp):
                    if yt < 100: return abs(yt - yp) <= 15
                    else: return abs(yt - yp) / yt <= 0.15
                acc_15 = (sum([check_15_15(yt, yp) for yt, yp in zip(y_test, y_pred)]) / len(y_test)) * 100
                
                # --- 결과 표시 ---
                
                # 1. 핵심 지표 (KPI)
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("MARD (오차율)", f"{mard:.2f}%", delta_color="inverse")
                kpi2.metric("15/15% 정확도", f"{acc_15:.2f}%")
                kpi3.metric("R-Squared", f"{r2:.4f}")
                kpi4.metric("샘플 수", f"{len(df)}개")
                
                st.divider()

                # 2. 인터랙티브 시계열 그래프 (Plotly)
                st.subheader("📈 실시간 혈당 추적")
                fig = go.Figure()
                # 실제 혈당
                fig.add_trace(go.Scatter(
                    y=y_test, mode='lines', name='실제 혈당 (Libre)',
                    line=dict(color='black', width=2)
                ))
                # 예측 혈당
                fig.add_trace(go.Scatter(
                    y=y_pred, mode='lines', name='AI 예측 (Predicted)',
                    line=dict(color='red', width=2, dash='dot')
                ))
                fig.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=20),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.05, x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 상세 분석 그래프 (2단)
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("##### 🎯 정확도 분석 (Zone A)")
                    fig_acc, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_test, y_pred, alpha=0.4, color='blue', s=30)
                    
                    # 기준선 및 Zone
                    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                    ax.plot([min_v, max_v], [min_v, max_v], 'k-', lw=1.5)
                    
                    x_rng = np.linspace(min_v, max_v, 100)
                    u_b = [x+15 if x<100 else x*1.15 for x in x_rng]
                    l_b = [x-15 if x<100 else x*0.85 for x in x_rng]
                    
                    ax.plot(x_rng, u_b, 'r--', lw=1)
                    ax.plot(x_rng, l_b, 'r--', lw=1)
                    ax.fill_between(x_rng, l_b, u_b, color='green', alpha=0.1, label='Zone A')
                    
                    ax.set_xlabel('Reference (mg/dL)')
                    ax.set_ylabel('Predicted (mg/dL)')
                    ax.legend(loc='upper left')
                    ax.grid(True, linestyle=':', alpha=0.6)
                    st.pyplot(fig_acc)
                    
                with c2:
                    st.markdown("##### 📊 오차 분포 (0에 가까울수록 좋음)")
                    errors = y_pred - y_test
                    fig_hist, ax2 = plt.subplots(figsize=(6, 5))
                    sns.histplot(errors, kde=True, bins=25, color='orange', ax=ax2)
                    ax2.axvline(0, color='black', linestyle='--')
                    ax2.set_xlabel('Error (mg/dL)')
                    ax2.set_ylabel('Frequency')
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig_hist)
                
                # 4. 엑셀 다운로드
                st.subheader("📥 리포트 다운로드")
                
                # 결과 데이터 정리
                res_df = df.copy()
                res_df['Predicted_Glucose'] = np.nan
                res_df.loc[y_test.index, 'Predicted_Glucose'] = y_pred
                res_df['Error'] = res_df['Predicted_Glucose'] - res_df['gl']
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Raw_Data')
                    summary = pd.DataFrame({
                        'Item': ['Experiment Memo', 'Lag Minutes', 'Warmup Hours', 'R2', 'MARD', '15/15 Accuracy'],
                        'Value': [memo, lag_min, warmup_hr, r2, f"{mard:.2f}%", f"{acc_15:.2f}%"]
                    })
                    summary.to_excel(writer, index=False, sheet_name='Summary')
                    
                st.download_button(
                    label="📊 엑셀 파일 받기 (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"AGMS_Result_{memo}.xlsx" if memo else "AGMS_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    else:
        st.warning("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")
else:
    st.info("👈 파일 업로드 후 '분석 실행' 버튼을 눌러주세요.")
