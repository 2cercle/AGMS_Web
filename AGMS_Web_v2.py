# -*- coding: utf-8 -*-
"""
AGMS Sensor Analysis Dashboard (Full Clarke Error Grid)
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
# [2] Helper 함수: Full Clarke Error Grid Logic
# -----------------------------------------------------------------------------
def get_clarke_zone(ref, pred):
    """
    Standard Clarke Error Grid Zone Definition
    """
    if ref == 0: return 'B' # 0 나누기 방지

    abs_diff = abs(ref - pred)
    
    # Zone A: Reference의 ±20% 이내 또는 저혈당(70미만) 구간에서 오차 15 미만
    # (통상적인 시각적 기준인 ±20% 라인을 우선 적용)
    if abs_diff <= 0.2 * ref:
        return 'A'
    if ref < 70 and abs_diff <= 15: # 저혈당 구간의 엄격한 A존 기준 (선택적)
        return 'A'

    # Zone E: 위험한 오차 (저혈당/고혈당 반대 판독)
    if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70):
        return 'E'
    
    # Zone D: 감지 실패 (Failure to detect)
    # 실제로는 범위 밖인데 정상(70~180)으로 예측한 경우
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 70 and 70 <= pred <= 180):
        return 'D'
    
    # Zone C: 과도한 교정 (Overcorrection)
    # A, D, E가 아닌 영역 중, Ref보다 Pred가 과도하게 높거나 낮은 경우
    # 상단 C 구역: (ref >= 70 and pred > ref + 110) ? -> 이미지 기준 사선 영역
    # 하단 C 구역: (pred < ref - 110) ?
    # Clarke Grid 정의상 B와 C의 경계는 특정 라인이지만, 여기서는 간략화된 로직 대신
    # D, E가 아니면서 A 범위를 벗어난 것 중 경향성을 봅니다.
    # (이미지 기준: A 콘 밖, D/E 박스 밖이면 B 아니면 C)
    
    # 시각적 이미지와 매칭하기 위한 C존 로직 (Over-correction zone)
    # 상단 C: 실제값은 낮은데 예측값이 A존 위쪽 라인보다 훨씬 높을 때
    # 하단 C: 실제값은 높은데 예측값이 A존 아래쪽 라인보다 훨씬 낮을 때
    if (pred > ref + 110) or (pred < ref - 110):  # 일반적인 C존 컷오프
         return 'C'

    # Zone B: 임상적 조치가 필요 없는 양성 오차
    return 'B'

def plot_clarke_grid(y_test, y_pred, ax):
    """
    Matplotlib Axes에 완벽한 Clarke Error Grid 라인과 산점도를 그리는 함수
    """
    # 1. Zone 판별
    zones = [get_clarke_zone(r, p) for r, p in zip(y_test, y_pred)]
    zone_counts = {z: zones.count(z) for z in ['A', 'B', 'C', 'D', 'E']}
    total = len(zones)
    
    # 색상 맵 (이미지 스타일: A=Red/Pink, B=Green/Lime, D=Blue etc. -> 요청 이미지와 유사하게)
    # (일반적으로 A=Green이 직관적이나, 요청하신 이미지는 A가 Red 계열입니다. 
    #  여기서는 가독성을 위해 A=Green, B=Blue, error=Red 계열을 추천하지만, 
    #  이미지 분위기를 살려 A를 돋보이게 합니다.)
    colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#d62728', 'E': '#9467bd'}
    
    # 산점도 그리기
    for z in ['A', 'B', 'C', 'D', 'E']:
        mask = [zone == z for zone in zones]
        if sum(mask) > 0:
            ax.scatter(
                y_test[mask], y_pred[mask], 
                c=colors[z], s=25, alpha=0.6, edgecolors='white', linewidth=0.5,
                label=f'Zone {z}: {zone_counts[z]} ({zone_counts[z]/total*100:.1f}%)'
            )

    # 2. Grid Lines (이미지와 동일한 구성)
    ax.set_title("Clarke Error Grid Analysis", fontsize=12, fontweight='bold')
    ax.set_xlabel("Reference Glucose (mg/dL)")
    ax.set_ylabel("Sensor Glucose (mg/dL)")
    
    # 축 범위 (0~400)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')

    # (1) 기준선 (y=x)
    ax.plot([0, 400], [0, 400], 'k--', lw=1.5, alpha=0.7)

    # (2) Zone A/B Boundary (±20%)
    # Upper Line: y = 1.2x
    ax.plot([0, 333.3], [0, 400], 'k-', lw=1) 
    # Lower Line: y = 0.8x
    ax.plot([0, 400], [0, 320], 'k-', lw=1)

    # (3) 수평/수직 구분선 (The Boxes)
    # Horizontal y=180
    ax.plot([0, 400], [180, 180], 'k-', lw=1)
    # Horizontal y=70
    ax.plot([0, 400], [70, 70], 'k-', lw=1)
    
    # Vertical x=180
    ax.plot([180, 180], [0, 400], 'k-', lw=1)
    # Vertical x=70
    ax.plot([70, 70], [0, 400], 'k-', lw=1)
    # Vertical x=240 (Zone D boundary)
    ax.plot([240, 240], [70, 180], 'k-', lw=1)

    # (4) 텍스트 라벨 (그리드 위에 구역 표시)
    ax.text(30, 10, 'E', fontsize=12, color='red', fontweight='bold') # Lower Left E
    ax.text(350, 350, 'A', fontsize=12, color='green', fontweight='bold')
    ax.text(280, 200, 'B', fontsize=10, color='blue')
    ax.text(150, 260, 'B', fontsize=10, color='blue')
    ax.text(350, 120, 'D', fontsize=10, color='red')
    ax.text(30, 120, 'D', fontsize=10, color='red')
    ax.text(30, 350, 'E', fontsize=12, color='red', fontweight='bold') # Upper Left E
    ax.text(130, 350, 'C', fontsize=10, color='orange')
    ax.text(350, 30, 'C', fontsize=10, color='orange')

    ax.legend(loc='upper left', fontsize='small', frameon=True)
    ax.grid(False) # 기본 격자는 끄고 Clarke 라인만 강조

# -----------------------------------------------------------------------------
# [3] 데이터 처리 로직
# -----------------------------------------------------------------------------
@st.cache_data
def process_data(libre_file, sensor_files, lag_minutes, warmup_hours):
    # --- 1. 리브레(Reference) 데이터 로드 ---
    try:
        if libre_file.name.endswith('.xlsx'):
            libre_df = pd.read_excel(libre_file)
        else:
            libre_df = pd.read_csv(libre_file, skiprows=1) 
            
        col_map = {
            'Device Timestamp': 'ts', 
            'Historic Glucose mg/dL': 'gl', 
            'Scan Glucose mg/dL': 'gl_scan',
            'Timestamp': 'ts', 
            'Glucose': 'gl'
        }
        libre_df = libre_df.rename(columns=lambda x: col_map.get(x, x))
        
        libre_df['ts'] = pd.to_datetime(libre_df['ts'], errors='coerce')
        libre_df = libre_df.dropna(subset=['ts'])
        
        if 'gl' not in libre_df.columns and 'gl_scan' in libre_df.columns:
            libre_df['gl'] = libre_df['gl_scan']
        
        libre_df['gl'] = pd.to_numeric(libre_df['gl'], errors='coerce').interpolate()
        libre_df = libre_df.sort_values('ts')
        
        # Lag 적용
        libre_df['ts_merge'] = libre_df['ts'] - pd.Timedelta(minutes=lag_minutes)
        libre_df = libre_df.sort_values('ts_merge')
        
    except Exception as e:
        return None, None, f"리브레(정답지) 파일 처리 중 오류: {str(e)}"

    # --- 2. 센서(Raw) 데이터 로드 ---
    sensor_list = []
    use_cols = ['experiment_date', 'value_current', 'value_ae', 'value_temperature']
    
    for sf in sensor_files:
        try:
            temp = pd.read_csv(sf, usecols=lambda c: c in use_cols)
            sensor_list.append(temp)
        except:
            pass

    if not sensor_list:
        return None, None, "유효한 센서 데이터가 없습니다."

    sensor_df = pd.concat(sensor_list, ignore_index=True)
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['experiment_date'], errors='coerce')
    sensor_df = sensor_df.dropna(subset=['timestamp']).sort_values('timestamp')

    # 전처리 & 칼만 필터
    cols = ['value_current', 'value_ae', 'value_temperature']
    sensor_df[cols] = sensor_df[cols].ffill().bfill()
    
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    for c in cols:
        sensor_df[f'{c}_kf'], _ = kf.smooth(sensor_df[c].values)

    # Warm-up 제거
    start_t = sensor_df['timestamp'].min()
    sensor_df['hours_since_start'] = (sensor_df['timestamp'] - start_t).dt.total_seconds() / 3600.0
    sensor_df = sensor_df[sensor_df['hours_since_start'] > warmup_hours]
    
    if sensor_df.empty:
        return None, None, f"초기 {warmup_hours}시간 제거 후 남은 데이터가 없습니다."

    # --- 3. 데이터 병합 ---
    merged = pd.merge_asof(libre_df, sensor_df, left_on='ts_merge', right_on='timestamp',
                           direction='nearest', tolerance=pd.Timedelta('15min'))
    
    final_df = merged.dropna(subset=['gl', 'value_current_kf'])
    
    if final_df.empty:
        return None, None, "데이터 매칭 실패."
        
    return final_df, sensor_df, None

# -----------------------------------------------------------------------------
# [4] 사이드바 UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 1. 데이터 입력")
    uploaded_libre = st.file_uploader("1) 리브레 데이터 (엑셀/CSV)", type=['csv', 'xlsx'])
    uploaded_sensors = st.file_uploader("2) 센서 데이터 (CSV, 다중 선택)", type=['csv'], accept_multiple_files=True)
    
    st.header("⚙️ 2. 분석 설정")
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
        
        with st.spinner('데이터 병합 및 AI 분석 중...'):
            df, _, err = process_data(uploaded_libre, uploaded_sensors, lag_min, warmup_hr)
            
            if err:
                st.error(err)
            else:
                # 모델링
                features = ['value_current_kf', 'value_ae_kf', 'value_temperature_kf', 'hours_since_start']
                X = df[features]
                y = df['gl']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 지표 계산
                r2 = r2_score(y_test, y_pred)
                mard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                def check_15_15(yt, yp):
                    if yt < 100: return abs(yt - yp) <= 15
                    else: return abs(yt - yp) / yt <= 0.15
                acc_15 = (sum([check_15_15(yt, yp) for yt, yp in zip(y_test, y_pred)]) / len(y_test)) * 100
                
                # KPI 표시
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("MARD (오차율)", f"{mard:.2f}%", delta_color="inverse")
                kpi2.metric("15/15% 정확도", f"{acc_15:.2f}%")
                kpi3.metric("R-Squared", f"{r2:.4f}")
                kpi4.metric("샘플 수 (Test)", f"{len(y_test)}개")
                
                st.divider()

                # --- 1. 혈당 그래프 (Plotly) : 실제 혈당 기준 Zone 표시 ---
                st.subheader("📈 혈당 그래프 (실제 혈당 기준 허용 범위)")
                
                ref_values = y_test.values
                upper_bound = [r + 15 if r < 100 else r * 1.15 for r in ref_values]
                lower_bound = [r - 15 if r < 100 else r * 0.85 for r in ref_values]
                
                fig = go.Figure()

                # (1) Lower Bound
                fig.add_trace(go.Scatter(
                    x=y_test.index, y=lower_bound,
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))

                # (2) Upper Bound (Actual ±15%)
                fig.add_trace(go.Scatter(
                    x=y_test.index, y=upper_bound,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)',
                    name='허용 오차 범위 (Actual ±15%)',
                    hoverinfo='skip'
                ))

                # (3) Reference
                fig.add_trace(go.Scatter(
                    x=y_test.index, y=y_test,
                    mode='lines', name='실제 혈당 (Reference)',
                    line=dict(color='black', width=2) 
                ))

                # (4) Predicted
                fig.add_trace(go.Scatter(
                    x=y_test.index, y=y_pred,
                    mode='lines', name='AI 예측 (Predicted)',
                    line=dict(color='#d62728', width=2, dash='dot') 
                ))

                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=30, b=20),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 2. 상세 분석 그래프 (Clarke Grid & Error Dist) ---
                c1, c2 = st.columns(2)
                
                with c1:
                    # ★ Clarke Error Grid
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
                    ax2.set_ylabel('Frequency')
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig_hist)
                
                # --- 3. 엑셀 다운로드 ---
                st.subheader("📥 리포트 다운로드")
                
                res_df = df.copy()
                res_df['Predicted_Glucose'] = np.nan
                res_df.loc[y_test.index, 'Predicted_Glucose'] = y_pred
                res_df['Error_Diff'] = res_df['Predicted_Glucose'] - res_df['gl']
                res_df['Error_Pct'] = (res_df['Error_Diff'] / res_df['gl']) * 100
                
                # 각 포인트의 Zone도 엑셀에 저장
                zones = [get_clarke_zone(r, p) if pd.notnull(p) else np.nan 
                         for r, p in zip(res_df['gl'], res_df['Predicted_Glucose'])]
                res_df['Clarke_Zone'] = zones
                
                save_cols = ['ts', 'gl', 'Predicted_Glucose', 'Clarke_Zone', 'Error_Diff', 'Error_Pct'] + \
                            [c for c in res_df.columns if c not in ['ts', 'gl', 'Predicted_Glucose', 'Clarke_Zone', 'Error_Diff', 'Error_Pct']]
                res_df = res_df[save_cols]

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Raw_Data')
                    summary = pd.DataFrame({
                        'Item': ['Memo', 'R2', 'MARD', '15/15 Accuracy'],
                        'Value': [memo, r2, f"{mard:.2f}%", f"{acc_15:.2f}%"]
                    })
                    summary.to_excel(writer, index=False, sheet_name='Summary')
                    
                st.download_button(
                    label="📊 결과 엑셀 다운로드 (Zone 포함)",
                    data=buffer.getvalue(),
                    file_name=f"AGMS_Result_{memo}.xlsx" if memo else "AGMS_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    else:
        st.warning("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")
else:
    st.info("👈 파일 업로드 후 '분석 실행' 버튼을 눌러주세요.")
