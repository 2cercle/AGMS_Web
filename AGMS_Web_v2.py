# -*- coding: utf-8 -*-
"""
AGMS Sensor Analysis Dashboard
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

# -----------------------------------------------------------------------------
# [1] 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AGMS 센서 분석 대시보드",
    page_icon="🩸",
    layout="wide"
)

# 시각화 한글 폰트 설정 (OS별 대응)
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# [2] 데이터 처리 로직 (캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_data
def process_data(libre_file, sensor_files, lag_minutes, use_auto_lag, warmup_hours):
    # 1. 리브레(정답지) 로드
    try:
        if libre_file.name.endswith('.xlsx'):
            libre_df = pd.read_excel(libre_file)
        else:
            libre_df = pd.read_csv(libre_file, skiprows=1) # 헤더 위치에 따라 조정 필요
            
        # 컬럼 매핑
        col_map = {'Device Timestamp': 'ts', 'Historic Glucose mg/dL': 'gl', 'Scan Glucose mg/dL': 'gl_scan'}
        libre_df = libre_df.rename(columns=lambda x: col_map.get(x, x))
        
        libre_df['ts'] = pd.to_datetime(libre_df['ts'], errors='coerce')
        libre_df = libre_df.dropna(subset=['ts'])
        
        if 'gl' not in libre_df.columns and 'gl_scan' in libre_df.columns:
            libre_df['gl'] = libre_df['gl_scan']
        
        libre_df['gl'] = pd.to_numeric(libre_df['gl'], errors='coerce').interpolate()
        libre_df = libre_df.sort_values('ts')
        
    except Exception as e:
        return None, None, f"리브레 파일 오류: {str(e)}", 0

    # 2. 센서 데이터 로드 (다중 파일)
    sensor_list = []
    use_cols = ['experiment_date', 'value_current', 'value_ae', 'value_temperature']
    
    for sf in sensor_files:
        try:
            temp = pd.read_csv(sf, usecols=lambda c: c in use_cols)
            sensor_list.append(temp)
        except: pass

    if not sensor_list:
        return None, None, "유효한 센서 데이터가 없습니다. (CSV 형식 확인)", 0

    sensor_df = pd.concat(sensor_list, ignore_index=True)
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['experiment_date'], errors='coerce')
    sensor_df = sensor_df.dropna(subset=['timestamp']).sort_values('timestamp')

    # 전처리: 결측치 & 칼만 필터
    cols = ['value_current', 'value_ae', 'value_temperature']
    sensor_df[cols] = sensor_df[cols].ffill().bfill()
    
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    for c in cols:
        sensor_df[f'{c}_kf'], _ = kf.smooth(sensor_df[c].values)

    # 초기 제거 (Warm-up)
    start_t = sensor_df['timestamp'].min()
    sensor_df['hours_since_start'] = (sensor_df['timestamp'] - start_t).dt.total_seconds() / 3600.0
    sensor_df = sensor_df[sensor_df['hours_since_start'] > warmup_hours]
    
    if sensor_df.empty:
        return None, None, "초기 제거 후 남은 데이터가 없습니다.", 0

    # 3. 시간 동기화 (Auto-Lag or Manual)
    final_lag = lag_minutes
    
    if use_auto_lag:
        # 상관관계 기반 최적 시간 찾기
        l_res = libre_df.set_index('ts')['gl'].resample('1T').mean().interpolate()
        s_res = sensor_df.set_index('timestamp')['value_current_kf'].resample('1T').mean().interpolate()
        
        common_idx = l_res.index.intersection(s_res.index)
        if len(common_idx) > 30:
            best_corr = 0
            # -120분 ~ +120분 탐색
            test_lags = range(-120, 121, 1)
            df_corr = pd.DataFrame({'gl': l_res, 'cur': s_res}).dropna()
            
            corrs = []
            for lag in test_lags:
                shifted_cur = df_corr['cur'].shift(-lag)
                # 혈당과 전류는 반비례 관계가 일반적이므로 절대값으로 비교하거나 음의 상관관계 확인
                corrs.append(abs(df_corr['gl'].corr(shifted_cur)))
            
            final_lag = test_lags[np.argmax(corrs)]
    
    # Lag 적용
    libre_df['ts_merge'] = libre_df['ts'] - pd.Timedelta(minutes=final_lag)
    libre_df = libre_df.sort_values('ts_merge')

    # 4. 병합
    merged = pd.merge_asof(libre_df, sensor_df, left_on='ts_merge', right_on='timestamp',
                           direction='nearest', tolerance=pd.Timedelta('15min'))
    
    final_df = merged.dropna(subset=['gl', 'value_current_kf'])
    
    if final_df.empty:
        return None, None, "데이터 매칭 실패. 시간 범위가 겹치지 않습니다.", 0
        
    return final_df, sensor_df, None, final_lag

# -----------------------------------------------------------------------------
# [3] 사이드바 UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 1. 파일 업로드")
    uploaded_libre = st.file_uploader("리브레 정답지 (CSV/Excel)", type=['csv', 'xlsx'])
    uploaded_sensors = st.file_uploader("센서 데이터 (CSV 다중선택)", type=['csv'], accept_multiple_files=True)
    
    st.header("⚙️ 2. 파라미터 조정")
    use_auto_lag = st.checkbox("최적 시간지연 자동 찾기", value=True, help="체크 시 AI가 상관분석을 통해 지연 시간을 자동 계산합니다.")
    lag_min = st.number_input("시간 지연 (분)", value=15, step=1, disabled=use_auto_lag)
    warmup_hr = st.number_input("초기 제거 (시간)", value=24, step=1)
    
    st.header("📝 3. 실험 조건")
    memo = st.text_input("실험 메모 (제목으로 표시됨)", placeholder="예: 24382 이동근, 카본 공정 A타입")
    
    st.divider()
    run_btn = st.button("분석 실행 🚀", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# [4] 메인 대시보드 UI
# -----------------------------------------------------------------------------
if run_btn:
    if uploaded_libre and uploaded_sensors:
        # 타이틀 설정
        report_title = f"AGMS 분석 리포트: {memo}" if memo else "AGMS 분석 리포트"
        st.title(report_title)
        
        with st.spinner('데이터 분석 및 모델링 중...'):
            df, _, err, found_lag = process_data(uploaded_libre, uploaded_sensors, lag_min, use_auto_lag, warmup_hr)
            
            if err:
                st.error(err)
            else:
                # 모델링
                features = ['value_current_kf', 'value_ae_kf', 'value_temperature_kf', 'hours_since_start']
                X = df[features]
                y = df['gl']
                
                # 시계열 순서 유지 분할
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 지표 계산
                r2 = r2_score(y_test, y_pred)
                mard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # 15/15% 정확도
                def check_15_15(yt, yp):
                    if yt < 100: return abs(yt - yp) <= 15
                    else: return abs(yt - yp) / yt <= 0.15
                acc_15 = (sum([check_15_15(yt, yp) for yt, yp in zip(y_test, y_pred)]) / len(y_test)) * 100
                
                # -------------------------
                # 상단 지표 (Metrics)
                # -------------------------
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MARD (오차율)", f"{mard:.2f}%", delta_color="inverse")
                m2.metric("15/15% 정확도", f"{acc_15:.2f}%")
                m3.metric("R-Squared", f"{r2:.4f}")
                m4.metric("분석 샘플 수", f"{len(df)}개")
                
                if use_auto_lag:
                    st.success(f"🤖 AI가 찾은 최적 시간 지연: **{found_lag}분** (적용됨)")
                
                st.divider()

                # -------------------------
                # 차트 및 분석 (Column Layout)
                # -------------------------
                
                # 1. 인터랙티브 그래프 (전체 너비)
                st.subheader("📈 실시간 혈당 추적 (Interactive)")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test, mode='lines', name='실제 혈당 (Libre)', line=dict(color='black', width=2)))
                fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='AI 예측 (Predicted)', line=dict(color='red', width=2, dash='dot')))
                fig.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. 하단 분석 그래프 (2단 분할)
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("🎯 정확도 분석 (Zone A)")
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
                    st.subheader("📊 오차 분포 (Histogram)")
                    errors = y_pred - y_test
                    fig_hist, ax2 = plt.subplots(figsize=(6, 5))
                    sns.histplot(errors, kde=True, bins=25, color='orange', ax=ax2)
                    ax2.axvline(0, color='black', linestyle='--')
                    ax2.set_xlabel('Error (mg/dL)')
                    ax2.set_ylabel('Frequency')
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig_hist)
                
                # -------------------------
                # 데이터 다운로드
                # -------------------------
                st.subheader("📥 데이터 내보내기")
                
                # 결과 DF 생성
                res_df = df.copy()
                # 테스트셋에 대한 예측값 매핑 (간단히 표시)
                res_df['Predicted_Glucose'] = np.nan
                # 인덱스 기준으로 매핑 (주의: train/test split시 인덱스가 섞이지 않았으므로 가능)
                res_df.loc[y_test.index, 'Predicted_Glucose'] = y_pred
                res_df['Error'] = res_df['Predicted_Glucose'] - res_df['gl']
                
                # 엑셀 변환
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Raw_Data')
                    # 요약 시트
                    summary = pd.DataFrame({
                        'Parameter': ['Memo', 'Lag Minutes', 'Warmup Hours', 'R2', 'MARD', '15/15 Accuracy'],
                        'Value': [memo, found_lag, warmup_hr, r2, f"{mard:.2f}%", f"{acc_15:.2f}%"]
                    })
                    summary.to_excel(writer, index=False, sheet_name='Summary')
                    
                st.download_button(
                    label="엑셀 리포트 다운로드 (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"AGMS_Report_{memo}.xlsx" if memo else "AGMS_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    else:
        st.warning("👈 왼쪽 사이드바에서 리브레 파일과 센서 파일을 업로드해주세요.")
else:
    # 초기 안내 화면
    st.info("👈 왼쪽 사이드바에서 분석 조건을 설정하고 '분석 실행'을 눌러주세요.")
    st.markdown("""
    ### 💡 사용 가이드
    1. **파일 업로드**: 리브레 엑셀/CSV 파일과 센서 데이터 CSV 파일들을 선택합니다.
    2. **파라미터**: 
        - **최적 시간지연 자동 찾기**: 체크하면 AI가 혈당 그래프 패턴을 보고 시간을 자동으로 맞춥니다. (정확도 향상 추천)
        - **초기 제거**: 센서 부착 직후 불안정한 데이터(Warm-up)를 제거할 시간을 입력합니다.
    3. **실행**: 분석 실행 버튼을 누르면 대시보드가 생성됩니다.
    """)