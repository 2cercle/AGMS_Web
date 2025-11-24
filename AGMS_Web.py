# -*- coding: utf-8 -*-
"""
AGMS Analysis Web Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AGMS 센서 분석 시스템",
    page_icon="🩸",
    layout="wide"
)

# 한글 폰트 설정 (Matplotlib용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 2. 데이터 처리 함수 (캐싱 적용으로 속도 최적화)
# -----------------------------------------------------------------------------
@st.cache_data
def process_agms_data(libre_file, sensor_files, lag_minutes, warmup_hours):
    # (1) 리브레 데이터 로드
    try:
        # 엑셀인지 CSV인지 구분
        if libre_file.name.endswith('.xlsx'):
            libre_df = pd.read_excel(libre_file)
        else:
            # 헤더 찾기 로직 (간소화)
            libre_df = pd.read_csv(libre_file, skiprows=1)

        # 컬럼명 통일
        col_map = {'Device Timestamp': 'ts', 'Historic Glucose mg/dL': 'gl', 'Scan Glucose mg/dL': 'gl_scan'}
        libre_df = libre_df.rename(columns=lambda x: col_map.get(x, x))
        
        libre_df['ts'] = pd.to_datetime(libre_df['ts'], errors='coerce')
        libre_df = libre_df.dropna(subset=['ts'])
        
        if 'gl' not in libre_df.columns and 'gl_scan' in libre_df.columns:
            libre_df['gl'] = libre_df['gl_scan']
        
        libre_df['gl'] = pd.to_numeric(libre_df['gl'], errors='coerce').interpolate()
        
        # 시간 지연(Lag) 보정 적용
        libre_df['ts_merge'] = libre_df['ts'] - pd.Timedelta(minutes=lag_minutes)
        libre_df = libre_df.sort_values('ts_merge')
        
    except Exception as e:
        return None, None, f"리브레 파일 오류: {str(e)}"

    # (2) 센서 데이터 로드 (다중 파일)
    sensor_list = []
    use_cols = ['experiment_date', 'value_current', 'value_ae', 'value_temperature']
    
    for sf in sensor_files:
        try:
            # CSV 로드
            temp = pd.read_csv(sf, usecols=lambda c: c in use_cols)
            sensor_list.append(temp)
        except:
            pass # 컬럼 안맞는 파일은 패스

    if not sensor_list:
        return None, None, "유효한 센서 데이터가 없습니다."

    sensor_df = pd.concat(sensor_list, ignore_index=True)
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['experiment_date'], errors='coerce')
    sensor_df = sensor_df.dropna(subset=['timestamp']).sort_values('timestamp')

    # 결측치 채우기
    cols = ['value_current', 'value_ae', 'value_temperature']
    sensor_df[cols] = sensor_df[cols].ffill().bfill()

    # 칼만 필터 (노이즈 제거)
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    for c in cols:
        sensor_df[f'{c}_kf'], _ = kf.smooth(sensor_df[c].values)

    # 초기 안정화 시간(Warm-up) 제거
    start_t = sensor_df['timestamp'].min()
    sensor_df['hours_since_start'] = (sensor_df['timestamp'] - start_t).dt.total_seconds() / 3600.0
    sensor_df = sensor_df[sensor_df['hours_since_start'] > warmup_hours]

    # (3) 데이터 병합
    merged = pd.merge_asof(libre_df, sensor_df, left_on='ts_merge', right_on='timestamp',
                           direction='nearest', tolerance=pd.Timedelta('15min'))
    
    final_df = merged.dropna(subset=['gl', 'value_current_kf'])
    
    if final_df.empty:
        return None, None, "매칭된 데이터가 없습니다. 시간 범위나 지연 설정을 확인하세요."

    return final_df, sensor_df, None

# -----------------------------------------------------------------------------
# 3. 사이드바 (입력 패널)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 분석 설정")
    
    st.header("1. 파일 업로드")
    uploaded_libre = st.file_uploader("리브레 정답지 (CSV/Excel)", type=['csv', 'xlsx'])
    uploaded_sensors = st.file_uploader("센서 데이터 (CSV 다중선택)", type=['csv'], accept_multiple_files=True)
    
    st.header("2. 파라미터 조정")
    lag_min = st.number_input("시간 지연 (분)", value=15, step=1, help="센서가 혈액보다 늦게 반응하는 시간")
    warmup_hr = st.number_input("초기 제거 (시간)", value=24, step=1, help="센서 부착 후 불안정한 초기 시간 제외")
    
    st.header("3. 실험 메모")
    memo_text = st.text_area("실험 조건 기록", placeholder="예: 24382 이동근, 카본 공정 A타입...")
    
    analyze_btn = st.button("분석 실행 🚀", type="primary")

# -----------------------------------------------------------------------------
# 4. 메인 화면 (출력 패널)
# -----------------------------------------------------------------------------
st.title("🩸 AGMS 연속혈당센서 성능 분석기")

if analyze_btn:
    if uploaded_libre and uploaded_sensors:
        with st.spinner('데이터 처리 및 AI 분석 중...'):
            df, raw_sensor, err_msg = process_agms_data(uploaded_libre, uploaded_sensors, lag_min, warmup_hr)
            
            if err_msg:
                st.error(err_msg)
            else:
                # --- 모델링 수행 ---
                features = ['value_current_kf', 'value_ae_kf', 'value_temperature_kf', 'hours_since_start']
                X = df[features]
                y = df['gl']
                
                # 시계열 순서대로 분할 (Shuffle=False)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # --- 정확도 지표 계산 ---
                r2 = r2_score(y_test, y_pred)
                
                # MARD
                mard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # 15/15% Accuracy
                def check_15_15(yt, yp):
                    if yt < 100: return abs(yt - yp) <= 15
                    else: return abs(yt - yp) / yt <= 0.15
                acc_count = sum([check_15_15(yt, yp) for yt, yp in zip(y_test, y_pred)])
                acc_15 = (acc_count / len(y_test)) * 100

                # --- 결과 대시보드 표시 ---
                
                # 1. 상단 메트릭
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MARD (오차율)", f"{mard:.2f}%", delta_color="inverse")
                col2.metric("15/15% 정확도", f"{acc_15:.2f}%")
                col3.metric("R-Squared", f"{r2:.4f}")
                col4.metric("데이터 샘플 수", f"{len(df)}개")
                
                if memo_text:
                    st.info(f"📝 **실험 메모:** {memo_text}")

                # 탭 구성
                tab1, tab2, tab3 = st.tabs(["📈 인터랙티브 그래프", "🎯 정확도 분석", "📥 데이터 다운로드"])

                # [Tab 1] 시계열 그래프 (Plotly 사용 - 줌/팬 가능)
                with tab1:
                    st.subheader("실시간 혈당 추적 (줌/팬 가능)")
                    
                    # Plotly 데이터 준비
                    plot_df = pd.DataFrame({
                        'Time': y_test.index, # 실제 시간축이 있다면 그것을 사용 권장
                        'Reference (Libre)': y_test.values,
                        'Prediction (AI)': y_pred
                    })
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(y=plot_df['Reference (Libre)'], name='실제 혈당(Libre)', line=dict(color='black', width=2)))
                    fig_ts.add_trace(go.Scatter(y=plot_df['Prediction (AI)'], name='예측 혈당(AI)', line=dict(color='red', dash='dot')))
                    
                    fig_ts.update_layout(
                        xaxis_title="샘플 포인트 (시간)",
                        yaxis_title="혈당 (mg/dL)",
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

                # [Tab 2] 정확도 분석 (Zone Plot)
                with tab2:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.subheader("Clarke Error Grid (유사)")
                        fig_sc, ax = plt.subplots(figsize=(6, 6))
                        ax.scatter(y_test, y_pred, alpha=0.4, color='blue')
                        
                        # 기준선
                        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                        ax.plot([min_v, max_v], [min_v, max_v], 'k-')
                        
                        # 15% Zone
                        x_line = np.linspace(min_v, max_v, 100)
                        upper = [x+15 if x<100 else x*1.15 for x in x_line]
                        lower = [x-15 if x<100 else x*0.85 for x in x_line]
                        
                        ax.plot(x_line, upper, 'r--', lw=1)
                        ax.plot(x_line, lower, 'r--', lw=1)
                        ax.fill_between(x_line, lower, upper, color='green', alpha=0.1, label='Zone A')
                        
                        ax.set_xlabel('Reference')
                        ax.set_ylabel('Predicted')
                        ax.legend()
                        st.pyplot(fig_sc)
                        
                    with col_b:
                        st.subheader("오차 분포 (Histogram)")
                        errors = y_pred - y_test
                        fig_hist, ax2 = plt.subplots(figsize=(6, 6))
                        sns.histplot(errors, kde=True, ax=ax2, color='orange')
                        ax2.axvline(0, color='k', linestyle='--')
                        ax2.set_xlabel('Error (mg/dL)')
                        st.pyplot(fig_hist)

                # [Tab 3] 데이터 다운로드
                with tab3:
                    st.subheader("분석 결과 데이터")
                    
                    # 결과 DataFrame 생성
                    result_df = df.copy()
                    # 테스트 셋 부분에만 예측값 할당 (나머지는 NaN)
                    # 실제 서비스에선 전체 예측을 할 수도 있음
                    result_df['Predicted_Glucose'] = np.nan
                    result_df.iloc[y_test.index, result_df.columns.get_loc('Predicted_Glucose')] = y_pred
                    
                    st.dataframe(result_df.head(100))
                    
                    # 엑셀 다운로드 버튼
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Analysis_Result')
                        # 요약 시트 추가
                        summary = pd.DataFrame({
                            'Metric': ['R2', 'MARD', '15/15 Accuracy', 'Memo'],
                            'Value': [r2, f"{mard:.2f}%", f"{acc_15:.2f}%", memo_text]
                        })
                        summary.to_excel(writer, index=False, sheet_name='Summary')
                        
                    st.download_button(
                        label="📥 엑셀 리포트 다운로드",
                        data=buffer.getvalue(),
                        file_name=f"AGMS_Analysis_Result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    else:
        st.warning("파일을 모두 업로드해주세요.")
else:
    st.info("👈 왼쪽 사이드바에서 파일을 업로드하고 '분석 실행' 버튼을 눌러주세요.")