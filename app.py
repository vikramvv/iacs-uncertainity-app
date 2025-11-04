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
    1e-6: {'ppm': 250, 'offset': 700e-12},      # 1 μA: 0.025% + 700 pA
    10e-6: {'ppm': 250, 'offset': 1e-9},        # 10 μA: 0.025% + 1 nA
    100e-6: {'ppm': 200, 'offset': 10e-9},      # 100 μA: 0.020% + 10 nA
    1e-3: {'ppm': 200, 'offset': 100e-9},       # 1 mA: 0.020% + 100 nA
    10e-3: {'ppm': 200, 'offset': 1e-6},        # 10 mA: 0.020% + 1 μA
    100e-3: {'ppm': 200, 'offset': 10e-6},      # 100 mA: 0.020% + 10 μA
    1.0: {'ppm': 500, 'offset': 500e-6},        # 1 A: 0.050% + 500 μA
    4.0: {'ppm': 1000, 'offset': 2.5e-3},       # 4 A: 0.100% + 2.5 mA
    5.0: {'ppm': 1000, 'offset': 2.5e-3},       # 5 A: 0.100% + 2.5 mA
    7.0: {'ppm': 1500, 'offset': 5e-3},         # 7 A: 0.150% + 5 mA
    10.0: {'ppm': 1500, 'offset': 5e-3},        # 10 A: 0.150% + 5 mA
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
    # If current exceeds all ranges, use the highest
    return ranges[-1], CURRENT_ACCURACY_TABLE[ranges[-1]]

def calculate_current_uncertainty(I: float) -> float:
    """Calculate current uncertainty based on specification table."""
    if I <= 0:
        return 0.0
    range_val, specs = get_current_range(I)
    ppm = specs['ppm']
    offset = specs['offset']
    # ΔI = (ppm/1e6) * I + offset
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
    # IACS = (rho_Cu * L) / (R * A) * 100
    return (RHO_CU * L_m) / (R * A_m2) * 100.0

def combined_R_uncertainty_abs(R: float, V: float, I: float, dV: float, dI: float) -> Optional[float]:
    """
    Calculate combined absolute uncertainty in R from voltmeter and ammeter.
    Since R = V/I, using error propagation:
    ΔR = R * √[(ΔV/V)² + (ΔI/I)²]
    """
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

def fmt_val_pm(val: Optional[float], pm: Optional[float], decimals: int = 3) -> str:
    """Format 'value ± uncertainty' if both exist."""
    if val is None or pm is None:
        return "–"
    if any(isinstance(x, float) and (math.isnan(x) or math.isinf(x)) for x in [val, pm]):
        return "–"
    return f"{val:.{decimals}f} ± {pm:.{decimals}f}"

def to_pp(val: Optional[float], decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "–"
    return f"{val:.{decimals}f}"

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="IACS & Uncertainty Calculator", layout="wide")
st.title("IACS & Uncertainty Calculator")
st.caption("Enter current and resistance (voltage calculated automatically). Uncertainties linked to instrument specifications.")

with st.sidebar:
    st.header("Display")
    precision = st.number_input("Display precision (decimals)", min_value=0, max_value=10, value=3, step=1)
    st.markdown("---")
    st.markdown("**Constants**")
    st.code(f"ρ_Cu (Ω·m) = {RHO_CU:.3e}")
    st.markdown("**Units**\n- R in Ω\n- L, d in mm\n- V in V, I in A")

# -----------------------------
# Inputs: measured values
# -----------------------------
st.subheader("📏 Measured Values")
col1, col2, col3 = st.columns(3)

with col1:
    I = st.number_input("Test Current I [A]", value=7.0, min_value=0.0, step=0.1, format="%.6f",
                       help="Current used for resistance measurement")
with col2:
    R = st.number_input("Resistance R [Ω]", value=0.012, min_value=0.0, step=0.000001, format="%.9f")
with col3:
    # Calculate voltage from I and R
    V = I * R if I > 0 and R > 0 else 0.0
    st.metric("Calculated Voltage V [V]", f"{V:.9f}")
    st.caption(f"V = I × R = {I:.6f} × {R:.9f}")

col4, col5 = st.columns(2)
with col4:
    L_mm = st.number_input("Length L [mm]", value=300.0, min_value=0.0, step=0.1, format="%.3f")
with col5:
    d_mm = st.number_input("Diameter d [mm]", value=2.5, min_value=0.0, step=0.001, format="%.6f")

# -----------------------------
# Inputs: absolute uncertainties
# -----------------------------
st.markdown("---")
st.subheader("🎯 Instrument Uncertainties")

col_unc1, col_unc2 = st.columns(2)

with col_unc1:
    st.markdown("**Length measurement:**")
    dL_option = st.selectbox("Length uncertainty", 
                             options=[0.01, 0.025, 0.05, 0.1],
                             index=2,
                             format_func=lambda x: f"±{x} mm")
    dL_mm = dL_option
    
    st.markdown("**Diameter measurement:**")
    dd_option = st.selectbox("Diameter uncertainty", 
                             options=[0.5, 2.0, 10.0, 20.0],
                             index=1,
                             format_func=lambda x: f"±{x} μm")
    dd_mm = dd_option / 1000.0  # Convert microns to mm

with col_unc2:
    st.markdown("**Voltmeter (2182A, 1 year):**")
    voltage_range = st.selectbox("Voltage range",
                                 options=['10mV', '100mV'],
                                 index=0 if V <= 0.01 else 1,
                                 help="Select range based on measured voltage")
    
    if voltage_range == '10mV':
        dV = 50e-9  # ±50 nV
        st.info("**Accuracy:** ±(50 ppm + 4 ppm of range) = ±50 nV for typical measurements")
    else:
        dV = 757e-9  # ±757 nV
        st.info("**Accuracy:** ±(30 ppm + 4 ppm of range) ≈ ±757 nV for ~12 mV")
    
    st.markdown("**Current source (2460, 1 year):**")
    # Auto-calculate current uncertainty
    dI = calculate_current_uncertainty(I)
    
    if I > 0:
        range_val, specs = get_current_range(I)
        ppm_pct = specs['ppm'] / 1e4  # Convert ppm to %
        offset_str = f"{specs['offset']*1e3:.1f} mA" if specs['offset'] >= 1e-3 else f"{specs['offset']*1e6:.1f} μA"
        st.info(f"**Range:** {range_val} A\n\n**Accuracy:** ±({ppm_pct:.3f}% + {offset_str}) = ±{dI*1e3:.3f} mA")
    else:
        st.warning("Enter current to calculate uncertainty")

# Calculate combined resistance uncertainty
dR = combined_R_uncertainty_abs(R, V, I, dV, dI)

if dR is not None:
    dR_pct = (dR / R) * 100
    st.success(f"**Combined resistance uncertainty:** ΔR = {dR:.9f} Ω ({dR_pct:.6f}%)")
else:
    st.warning("Cannot calculate ΔR - check that V, I, and R are all > 0")

# Geometry & IACS
A_mm2 = area_from_d(d_mm)
dA_mm2 = delta_A_from_d(A_mm2, d_mm, dd_mm)
IACS = iacs_percent(R, L_mm, A_mm2)

# IACS uncertainties
rel_u = relative_u_iacs(L_mm, dL_mm, R, dR if dR else float("nan"), A_mm2, dA_mm2)
abs_u = None if (IACS is None or rel_u is None) else IACS * rel_u  # absolute uncertainty in IACS units (pp)

# Calculate individual contributions to uncertainty
contrib_L = None
contrib_R = None
contrib_A = None
if rel_u is not None and rel_u > 0 and dR is not None:
    term_L = (dL_mm / L_mm) ** 2
    term_R = (dR / R) ** 2
    term_A = (dA_mm2 / A_mm2) ** 2
    total_var = term_L + term_R + term_A
    contrib_L = (term_L / total_var) * 100 if total_var > 0 else 0
    contrib_R = (term_R / total_var) * 100 if total_var > 0 else 0
    contrib_A = (term_A / total_var) * 100 if total_var > 0 else 0

# -----------------------------
# Validation
# -----------------------------
errors = []
if R <= 0: errors.append("Resistance R must be > 0.")
if L_mm <= 0: errors.append("Length L must be > 0.")
if d_mm <= 0: errors.append("Diameter d must be > 0.")
if I <= 0: errors.append("Current I must be > 0.")
for e in errors:
    st.error(e)

# -----------------------------
# Results (clean & focused)
# -----------------------------
if not errors:
    st.markdown("---")
    st.subheader("📊 Results")

    # Main results
    cA, cB = st.columns(2)
    with cA:
        st.metric("IACS [%]", to_pp(IACS, precision))
        st.metric("Absolute uncertainty in IACS [pp]", to_pp(abs_u, precision))
    with cB:
        st.metric("Relative uncertainty in IACS [%]", to_pp(rel_u * 100 if rel_u else None, precision))
        st.metric("Area A [mm²]", to_pp(A_mm2, precision))

    st.success(f"**Final result:** IACS = {fmt_val_pm(IACS, abs_u, precision)} % (percentage points)")

    # Detailed uncertainties
    with st.expander("📊 Detailed uncertainty breakdown", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ΔR [Ω]", f"{dR:.9f}" if dR else "–")
            st.caption(f"From ΔV and ΔI")
        with col2:
            st.metric("ΔL [mm]", f"{dL_mm:.3f}")
            st.caption(f"Selected: ±{dL_option} mm")
        with col3:
            st.metric("Δd [mm]", f"{dd_mm:.6f}")
            st.caption(f"Selected: ±{dd_option} μm")
        with col4:
            st.metric("ΔA [mm²]", to_pp(dA_mm2, precision + 3))
            st.caption("From Δd (doubled)")

        st.markdown("---")
        st.markdown("**Detailed instrument uncertainties:**")
        inst_col1, inst_col2 = st.columns(2)
        with inst_col1:
            st.metric("ΔV [nV]", f"{dV*1e9:.1f}")
            st.caption(f"Range: {voltage_range}")
        with inst_col2:
            st.metric("ΔI [mA]", f"{dI*1e3:.3f}")
            if I > 0:
                range_val, _ = get_current_range(I)
                st.caption(f"Range: {range_val} A")

        # Contribution chart
        if all(c is not None for c in [contrib_L, contrib_R, contrib_A]):
            st.markdown("---")
            st.markdown("**Uncertainty contributions to total IACS uncertainty:**")
            contrib_data = pd.DataFrame({
                'Source': ['Length (L)', 'Resistance (R)', 'Area (A=πd²/4)'],
                'Contribution [%]': [contrib_L, contrib_R, contrib_A]
            })
            st.dataframe(contrib_data, use_container_width=True, hide_index=True)
            
            # Show which is dominant
            max_contrib = max(contrib_L, contrib_R, contrib_A)
            if max_contrib == contrib_L:
                dominant = "Length"
            elif max_contrib == contrib_R:
                dominant = "Resistance"
            else:
                dominant = "Diameter/Area"
            st.info(f"**Dominant uncertainty source:** {dominant} ({max_contrib:.1f}%)")

# -----------------------------
# Multi-trial table with persistent state
# -----------------------------
st.markdown("---")
st.subheader("📋 Multi-trial comparison")

# Initialize or fetch trials dataframe in session state
if "trials_df" not in st.session_state:
    st.session_state.trials_df = pd.DataFrame(columns=[
        "I_A", "R_ohm", "L_mm", "d_mm", "dL_mm", "dd_um", "V_range"
    ])

# Button to push current inputs into trials
if st.button("➕ Add current inputs to trials"):
    new_row = pd.DataFrame([{
        "I_A": I,
        "R_ohm": R,
        "L_mm": L_mm,
        "d_mm": d_mm,
        "dL_mm": dL_option,
        "dd_um": dd_option,
        "V_range": voltage_range
    }])
    st.session_state.trials_df = pd.concat([st.session_state.trials_df, new_row], ignore_index=True)
    st.success("Added current inputs to trials.")

# Editable trials table
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

# Save edits back to state
st.session_state.trials_df = edited.copy()

# Compute results for all trials
if st.button("🔄 Compute for all trials") and len(st.session_state.trials_df) > 0:
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

        # Calculate V
        V_i = I_i * R_i
        
        # Convert units
        dd_i = dd_um_i / 1000.0  # μm to mm
        
        # Get voltage uncertainty based on range
        if V_range_i == '10mV':
            dV_i = 50e-9
        else:
            dV_i = 757e-9
        
        # Calculate current uncertainty
        dI_i = calculate_current_uncertainty(I_i)
        
        # Calculate combined R uncertainty
        dR_i = combined_R_uncertainty_abs(R_i, V_i, I_i, dV_i, dI_i)
        
        if dR_i is None:
            continue

        # Area, dA, IACS, uncertainties
        A_i = area_from_d(d_i)
        dA_i = delta_A_from_d(A_i, d_i, dd_i)
        IACS_i = iacs_percent(R_i, L_i, A_i)
        rel_u_i = relative_u_iacs(L_i, dL_i, R_i, dR_i, A_i, dA_i)
        abs_u_i = None if (IACS_i is None or rel_u_i is None) else IACS_i * rel_u_i

        results.append({
            "Trial": idx + 1,
            "I_A": I_i,
            "R_ohm": R_i,
            "V_V": V_i,
            "L_mm": L_i,
            "d_mm": d_i,
            "IACS_%": IACS_i,
            "abs_unc_pp": abs_u_i,
            "rel_unc_%": rel_u_i * 100 if rel_u_i else None,
            "dI_mA": dI_i * 1e3,
            "dV_nV": dV_i * 1e9,
            "dR_ohm": dR_i,
            "A_mm2": A_i
        })

    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df.style.format({
            "I_A": "{:.6f}",
            "R_ohm": "{:.9f}",
            "V_V": "{:.9f}",
            "L_mm": "{:.3f}",
            "d_mm": "{:.6f}",
            "IACS_%": "{:.3f}",
            "abs_unc_pp": "{:.3f}",
            "rel_unc_%": "{:.3f}",
            "dI_mA": "{:.3f}",
            "dV_nV": "{:.1f}",
            "dR_ohm": "{:.9f}",
            "A_mm2": "{:.6f}"
        }), use_container_width=True)

        # Export buttons
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📥 Download CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="iacs_results.csv",
                mime="text/csv"
            )
        with col_exp2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                res_df.to_excel(writer, index=False, sheet_name="results")
                st.session_state.trials_df.to_excel(writer, index=False, sheet_name="inputs")
            st.download_button(
                label="📥 Download Excel (.xlsx)",
                data=buf.getvalue(),
                file_name="iacs_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No valid trials to compute. Check your input data.")

# -----------------------------
# Help & Formulas
# -----------------------------
with st.expander("📖 Formulas & Notes"):
    st.markdown(r"""
**IACS (International Annealed Copper Standard):**
$$
\text{IACS} = \frac{\rho_{\text{Cu}} \cdot L}{R \cdot A} \times 100
$$
where $\rho_{\text{Cu}} = 1.724\times10^{-8}\ \Omega\cdot m$, $L$ in meters, $A$ in m².

**Voltage calculation:**
$$
V = I \times R
$$

**Area and its uncertainty:**
$$
A = \pi\frac{d^2}{4}, \quad \frac{\Delta A}{A} = 2\frac{\Delta d}{d} \Rightarrow \Delta A = A \cdot \frac{2\Delta d}{d}
$$

**Combined resistance uncertainty:**
Since $R = V/I$, using error propagation with absolute uncertainties:
$$
\Delta R = R \cdot \sqrt{\left(\frac{\Delta V}{V}\right)^2 + \left(\frac{\Delta I}{I}\right)^2}
$$

**Total relative uncertainty in IACS:**
$$
\frac{u_{\text{IACS}}}{\text{IACS}} = \sqrt{\left(\frac{\Delta L}{L}\right)^2 + \left(\frac{\Delta R}{R}\right)^2 + \left(\frac{\Delta A}{A}\right)^2}
$$

**Current uncertainty (Keithley 2460, 1 year):**
Based on current range, calculated as: $\Delta I = (\text{ppm}/10^6) \times I + \text{offset}$

**Voltage uncertainty (Keithley 2182A, 1 year):**
- 10 mV range: ±50 nV (50 ppm + 4 ppm of range)
- 100 mV range: ±757 nV (30 ppm + 4 ppm of range)

**Note:** The diameter uncertainty has double weight because area depends on $d^2$.
""")

# Show current accuracy table
with st.expander("📋 Current Accuracy Reference Table (Keithley 2460)"):
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
        
        example_calc = (specs['ppm'] / 1e6) * range_val + specs['offset']
        if example_calc >= 1e-3:
            calc_str = f"±{example_calc*1e3:.3f} mA"
        elif example_calc >= 1e-6:
            calc_str = f"±{example_calc*1e6:.3f} μA"
        else:
            calc_str = f"±{example_calc*1e9:.3f} nA"
        
        accuracy_ref.append({
            "Range": f"{range_val} A",
            "Accuracy": f"±({ppm_pct:.3f}% + {offset_str})",
            "Example": calc_str
        })
    
    st.dataframe(pd.DataFrame(accuracy_ref), use_container_width=True, hide_index=True)