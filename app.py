import math
import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants
# -----------------------------
RHO_CU = 1.724e-8  # Ω·m (IACS reference copper resistivity)

# Current accuracy specifications (2460 Source, 1 Year, 23°C ±5°C)
CURRENT_ACCURACY_TABLE = {
    1e-6: {'ppm': 250, 'offset': 700e-12},
    10e-6: {'ppm': 250, 'offset': 1e-9},
    100e-6: {'ppm': 200, 'offset': 10e-9},
    1e-3: {'ppm': 200, 'offset': 100e-9},
    10e-3: {'ppm': 200, 'offset': 1e-6},
    100e-3: {'ppm': 200, 'offset': 10e-6},
    1.0: {'ppm': 500, 'offset': 500e-6},
    4.0: {'ppm': 1000, 'offset': 2.5e-3},
    5.0: {'ppm': 1000, 'offset': 2.5e-3},
    7.0: {'ppm': 1500, 'offset': 5e-3},
    10.0: {'ppm': 1500, 'offset': 5e-3},
}

# -----------------------------
# Helpers & Calculations
# -----------------------------
def get_current_range(I: float) -> Tuple[float, dict]:
    """Get the appropriate current range and its accuracy specs."""
    ranges = sorted(CURRENT_ACCURACY_TABLE.keys())
    for range_val in ranges:
        if I <= range_val:
            return range_val, CURRENT_ACCURACY_TABLE[range_val]
    return ranges[-1], CURRENT_ACCURACY_TABLE[ranges[-1]]

def calculate_current_uncertainty(I: float) -> float:
    """Calculate current uncertainty based on specification table."""
    if I <= 0:
        return 0.0
    range_val, specs = get_current_range(I)
    ppm = specs['ppm']
    offset = specs['offset']
    return (ppm / 1e6) * I + offset

def area_from_d(d_mm: float) -> float:
    """Area in mm^2 from diameter in mm."""
    return math.pi * (d_mm ** 2) / 4.0

def delta_A_from_d(A_mm2: float, d_mm: float, delta_d_mm: float) -> float:
    """ΔA in mm^2 from Δd using ΔA/A = 2·Δd/d."""
    if d_mm <= 0 or A_mm2 <= 0:
        return float("nan")
    return A_mm2 * 2.0 * delta_d_mm / d_mm

def iacs_percent(R: float, L_mm: float, A_mm2: float) -> Optional[float]:
    """Compute IACS (%) given R (Ω), L (mm), A (mm^2)."""
    if R <= 0 or L_mm <= 0 or A_mm2 <= 0:
        return None
    L_m = L_mm / 1000.0
    A_m2 = A_mm2 / 1e6
    return (RHO_CU * L_m) / (R * A_m2) * 100.0

def combined_R_uncertainty_abs(R: float, V: float, I: float, dV: float, dI: float) -> Optional[float]:
    """Calculate combined absolute uncertainty in R from voltmeter and ammeter."""
    if V <= 0 or I <= 0 or R <= 0:
        return None
    rel_V = dV / V
    rel_I = dI / I
    return R * math.sqrt(rel_V**2 + rel_I**2)

def relative_u_iacs(L_mm: float, dL_mm: float,
                    R: float, dR: float,
                    A_mm2: float, dA_mm2: float) -> Optional[float]:
    """Relative uncertainty in IACS as a fraction."""
    if any(v is None for v in [L_mm, dL_mm, R, dR, A_mm2, dA_mm2]):
        return None
    if L_mm <= 0 or R <= 0 or A_mm2 <= 0:
        return None
    try:
        term_L = (dL_mm / L_mm) ** 2
        term_R = (dR / R) ** 2
        term_A = (dA_mm2 / A_mm2) ** 2
        return math.sqrt(term_L + term_R + term_A)
    except Exception:
        return None

def to_pp(val: Optional[float], decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "–"
    return f"{val:.{decimals}f}"

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IACS Calculator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact layout
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {margin-bottom: 0.5rem;}
    .stMetric {background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ IACS & Uncertainty Calculator")

# -----------------------------
# Sidebar: All Uncertainties & Settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    precision = st.number_input("Display precision", min_value=0, max_value=10, value=3, step=1)
    
    st.markdown("---")
    st.header("🎯 Uncertainties")
    
    st.subheader("Geometry")
    dL_option = st.selectbox("Length ΔL", 
                             options=[0.01, 0.025, 0.05, 0.1],
                             index=2,
                             format_func=lambda x: f"±{x} mm")
    dL_mm = dL_option
    
    dd_option = st.selectbox("Diameter Δd", 
                             options=[0.5, 2.0, 10.0, 20.0],
                             index=1,
                             format_func=lambda x: f"±{x} μm")
    dd_mm = dd_option / 1000.0
    
    st.subheader("Instruments")
    voltage_range = st.selectbox("Voltmeter range (2182A)",
                                 options=['10mV', '100mV'],
                                 index=0,
                                 help="Select based on measured voltage")
    
    if voltage_range == '10mV':
        dV = 50e-9
        st.caption("✓ ±50 nV")
    else:
        dV = 757e-9
        st.caption("✓ ±757 nV")
    
    st.caption("**Current source (2460):** Auto-calculated")
    
    st.markdown("---")
    st.markdown(f"**ρ_Cu** = {RHO_CU:.3e} Ω·m")

# -----------------------------
# Main Layout: Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Calculator", "📋 Multi-Trial", "📖 Reference"])

# =============================================================================
# TAB 1: CALCULATOR
# =============================================================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # LEFT COLUMN: Inputs
    with col_left:
        st.subheader("📥 Measured Values")
        
        col1a, col1b = st.columns(2)
        with col1a:
            I = st.number_input("Current I [A]", value=7.0, min_value=0.0, step=0.1, format="%.6f")
        with col1b:
            R = st.number_input("Resistance R [Ω]", value=0.012, min_value=0.0, step=0.000001, format="%.9f")
        
        # Calculate voltage
        V = I * R if I > 0 and R > 0 else 0.0
        st.info(f"**Calculated Voltage:** V = {V:.9f} V")
        
        col2a, col2b = st.columns(2)
        with col2a:
            L_mm = st.number_input("Length L [mm]", value=300.0, min_value=0.0, step=0.1, format="%.3f")
        with col2b:
            d_mm = st.number_input("Diameter d [mm]", value=2.5, min_value=0.0, step=0.001, format="%.6f")
        
        # Validation
        errors = []
        if R <= 0: errors.append("⚠️ Resistance R must be > 0")
        if L_mm <= 0: errors.append("⚠️ Length L must be > 0")
        if d_mm <= 0: errors.append("⚠️ Diameter d must be > 0")
        if I <= 0: errors.append("⚠️ Current I must be > 0")
        
        if errors:
            for e in errors:
                st.error(e)
    
    # RIGHT COLUMN: Results
    with col_right:
        st.subheader("📈 Results")
        
        if not errors:
            # Calculate everything
            dI = calculate_current_uncertainty(I)
            dR = combined_R_uncertainty_abs(R, V, I, dV, dI)
            A_mm2 = area_from_d(d_mm)
            dA_mm2 = delta_A_from_d(A_mm2, d_mm, dd_mm)
            IACS = iacs_percent(R, L_mm, A_mm2)
            rel_u = relative_u_iacs(L_mm, dL_mm, R, dR if dR else float("nan"), A_mm2, dA_mm2)
            abs_u = None if (IACS is None or rel_u is None) else IACS * rel_u
            
            # Main result - BIG
            st.markdown("### 🎯 IACS Result")
            if IACS and abs_u:
                st.markdown(f"<h2 style='color: #0066cc;'>{IACS:.{precision}f} ± {abs_u:.{precision}f} %</h2>", 
                           unsafe_allow_html=True)
            else:
                st.markdown("<h2>–</h2>", unsafe_allow_html=True)
            
            # Key metrics in compact grid
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Relative unc.", f"{rel_u * 100:.{precision}f} %" if rel_u else "–")
                st.metric("Area A", f"{A_mm2:.{precision}f} mm²")
            with m2:
                st.metric("ΔR", f"{dR:.9f} Ω" if dR else "–")
                st.metric("ΔI", f"{dI*1e3:.3f} mA")
            
            # Uncertainty breakdown
            st.markdown("---")
            st.markdown("**Uncertainty Breakdown**")
            
            # Calculate contributions
            contrib_L = contrib_R = contrib_A = None
            if rel_u is not None and rel_u > 0 and dR is not None:
                term_L = (dL_mm / L_mm) ** 2
                term_R = (dR / R) ** 2
                term_A = (dA_mm2 / A_mm2) ** 2
                total_var = term_L + term_R + term_A
                if total_var > 0:
                    contrib_L = (term_L / total_var) * 100
                    contrib_R = (term_R / total_var) * 100
                    contrib_A = (term_A / total_var) * 100
            
            if all(c is not None for c in [contrib_L, contrib_R, contrib_A]):
                contrib_data = pd.DataFrame({
                    'Source': ['Length', 'Resistance', 'Area (d²)'],
                    'Contribution %': [contrib_L, contrib_R, contrib_A]
                })
                st.dataframe(contrib_data, use_container_width=True, hide_index=True)
                
                # Dominant source
                max_contrib = max(contrib_L, contrib_R, contrib_A)
                if max_contrib == contrib_L:
                    dominant = "Length"
                elif max_contrib == contrib_R:
                    dominant = "Resistance"
                else:
                    dominant = "Diameter/Area"
                st.success(f"**Dominant:** {dominant} ({max_contrib:.1f}%)")
            
            # Detailed values in expander
            with st.expander("📊 Detailed Values", expanded=False):
                detail_data = {
                    "Parameter": ["ΔV", "ΔI", "ΔR", "ΔL", "Δd", "ΔA"],
                    "Value": [
                        f"{dV*1e9:.1f} nV",
                        f"{dI*1e3:.3f} mA",
                        f"{dR:.9f} Ω" if dR else "–",
                        f"{dL_mm:.3f} mm",
                        f"{dd_mm:.6f} mm ({dd_option} μm)",
                        f"{dA_mm2:.6f} mm²"
                    ]
                }
                st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: MULTI-TRIAL
# =============================================================================
with tab2:
    st.subheader("📋 Multi-Trial Comparison")
    
    # Initialize state
    if "trials_df" not in st.session_state:
        st.session_state.trials_df = pd.DataFrame(columns=[
            "I_A", "R_ohm", "L_mm", "d_mm", "dL_mm", "dd_um", "V_range"
        ])
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("➕ Add Current", use_container_width=True):
            new_row = pd.DataFrame([{
                "I_A": I, "R_ohm": R, "L_mm": L_mm, "d_mm": d_mm,
                "dL_mm": dL_option, "dd_um": dd_option, "V_range": voltage_range
            }])
            st.session_state.trials_df = pd.concat([st.session_state.trials_df, new_row], ignore_index=True)
            st.rerun()
    
    with col_btn2:
        compute_btn = st.button("🔄 Compute All", use_container_width=True, type="primary")
    
    # Editable table
    edited = st.data_editor(
        st.session_state.trials_df,
        num_rows="dynamic",
        use_container_width=True,
        key="trials_editor",
        column_config={
            "I_A": st.column_config.NumberColumn("I [A]", format="%.6f"),
            "R_ohm": st.column_config.NumberColumn("R [Ω]", format="%.9f"),
            "L_mm": st.column_config.NumberColumn("L [mm]", format="%.3f"),
            "d_mm": st.column_config.NumberColumn("d [mm]", format="%.6f"),
            "dL_mm": st.column_config.NumberColumn("ΔL [mm]", format="%.3f"),
            "dd_um": st.column_config.NumberColumn("Δd [μm]", format="%.1f"),
            "V_range": st.column_config.SelectboxColumn("V range", options=['10mV', '100mV'])
        }
    )
    st.session_state.trials_df = edited.copy()
    
    # Compute results
    if compute_btn and len(st.session_state.trials_df) > 0:
        results = []
        for idx, row in st.session_state.trials_df.iterrows():
            try:
                I_i = float(row.get("I_A", np.nan))
                R_i = float(row.get("R_ohm", np.nan))
                L_i = float(row.get("L_mm", np.nan))
                d_i = float(row.get("d_mm", np.nan))
                dL_i = float(row.get("dL_mm", np.nan))
                dd_um_i = float(row.get("dd_um", np.nan))
                V_range_i = str(row.get("V_range", "10mV"))
                
                if any(math.isnan(x) or x <= 0 for x in [I_i, R_i, L_i, d_i]):
                    continue
            except Exception:
                continue

            V_i = I_i * R_i
            dd_i = dd_um_i / 1000.0
            dV_i = 50e-9 if V_range_i == '10mV' else 757e-9
            dI_i = calculate_current_uncertainty(I_i)
            dR_i = combined_R_uncertainty_abs(R_i, V_i, I_i, dV_i, dI_i)
            
            if dR_i is None:
                continue

            A_i = area_from_d(d_i)
            dA_i = delta_A_from_d(A_i, d_i, dd_i)
            IACS_i = iacs_percent(R_i, L_i, A_i)
            rel_u_i = relative_u_iacs(L_i, dL_i, R_i, dR_i, A_i, dA_i)
            abs_u_i = None if (IACS_i is None or rel_u_i is None) else IACS_i * rel_u_i

            results.append({
                "Trial": idx + 1,
                "IACS_%": IACS_i,
                "±_pp": abs_u_i,
                "I_A": I_i,
                "R_ohm": R_i,
                "V_V": V_i,
                "L_mm": L_i,
                "d_mm": d_i,
                "A_mm2": A_i
            })

        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.format({
                "IACS_%": "{:.3f}",
                "±_pp": "{:.3f}",
                "I_A": "{:.6f}",
                "R_ohm": "{:.9f}",
                "V_V": "{:.9f}",
                "L_mm": "{:.3f}",
                "d_mm": "{:.6f}",
                "A_mm2": "{:.6f}"
            }), use_container_width=True)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "📥 CSV",
                    data=res_df.to_csv(index=False).encode("utf-8"),
                    file_name="iacs_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_d2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    res_df.to_excel(writer, index=False, sheet_name="results")
                    st.session_state.trials_df.to_excel(writer, index=False, sheet_name="inputs")
                st.download_button(
                    "📥 Excel",
                    data=buf.getvalue(),
                    file_name="iacs_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.warning("No valid trials to compute.")

# =============================================================================
# TAB 3: REFERENCE
# =============================================================================
with tab3:
    ref_col1, ref_col2 = st.columns(2)
    
    with ref_col1:
        st.subheader("📐 Formulas")
        st.markdown(r"""
**IACS Calculation:**
$$\text{IACS} = \frac{\rho_{\text{Cu}} \cdot L}{R \cdot A} \times 100$$

**Voltage:**
$$V = I \times R$$

**Area & Uncertainty:**
$$A = \pi\frac{d^2}{4}$$
$$\frac{\Delta A}{A} = 2\frac{\Delta d}{d}$$

**Resistance Uncertainty:**
$$\Delta R = R \cdot \sqrt{\left(\frac{\Delta V}{V}\right)^2 + \left(\frac{\Delta I}{I}\right)^2}$$

**Total IACS Uncertainty:**
$$\frac{u_{\text{IACS}}}{\text{IACS}} = \sqrt{\left(\frac{\Delta L}{L}\right)^2 + \left(\frac{\Delta R}{R}\right)^2 + \left(\frac{\Delta A}{A}\right)^2}$$
""")
    
    with ref_col2:
        st.subheader("🔌 Instrument Specs")
        
        st.markdown("**Keithley 2182A Nanovoltmeter** (1 year, 23°C ±5°C)")
        volt_specs = pd.DataFrame({
            "Range": ["10 mV", "100 mV"],
            "Accuracy": ["±(50 ppm + 4 ppm of range)", "±(30 ppm + 4 ppm of range)"],
            "Typical": ["±50 nV", "±757 nV"]
        })
        st.dataframe(volt_specs, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("**Keithley 2460 Current Source** (1 year, 23°C ±5°C)")
        accuracy_ref = []
        for range_val, specs in sorted(CURRENT_ACCURACY_TABLE.items()):
            ppm_pct = specs['ppm'] / 1e4
            if specs['offset'] >= 1e-3:
                offset_str = f"{specs['offset']*1e3:.1f} mA"
            elif specs['offset'] >= 1e-6:
                offset_str = f"{specs['offset']*1e6:.1f} μA"
            elif specs['offset'] >= 1e-9:
                offset_str = f"{specs['offset']*1e9:.1f} nA"
            else:
                offset_str = f"{specs['offset']*1e12:.1f} pA"
            
            accuracy_ref.append({
                "Range": f"{range_val} A",
                "Accuracy": f"±({ppm_pct:.3f}% + {offset_str})"
            })
        
        st.dataframe(pd.DataFrame(accuracy_ref), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("IACS Calculator v2.0 | All uncertainties based on manufacturer specifications")