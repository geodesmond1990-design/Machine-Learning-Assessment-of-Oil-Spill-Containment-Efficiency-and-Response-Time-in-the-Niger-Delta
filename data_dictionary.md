# Data Dictionary: Niger Delta Oil Spill Incident Database

**Source:** National Oil Spill Detection and Response Agency (NOSDRA), Abuja, Nigeria  
**Dataset:** oils_data.csv  
**Records:** 335 confirmed oil spill incidents  
**Period:** 22 January 2016 to 8 October 2024  
**Operators:** NAOC (Nigerian Agip Oil Company, ENI subsidiary) and SPDC (Shell Petroleum Development Company)

---

## Variable Definitions

| Column Name | Data Type | Description | Valid Values / Notes |
|-------------|-----------|-------------|----------------------|
| FID | Integer | Shapefile feature ID | Auto-incremented integer |
| Spill_ID | String | Unique incident identifier | Format: NOSDRA-XXXXXX |
| Status | String | Current incident status | Typically "Closed" for confirmed spills |
| Company | String | Operating company responsible | NAOC, SPDC |
| Incident_n | String | NOSDRA incident reference number | — |
| Incident_d | Date | Date incident occurred | Format: YYYY-MM-DD; used as primary date |
| Report_dat | Date | Date incident reported to NOSDRA | Should be within 24h of Incident_d per NOSDRA regulations; 30 records missing |
| Contaminan | String | Type of contaminant released | cr=crude oil; co=condensate; no=produced water (formation brine); other categories: nil, none, unknown |
| Estimated | Float | Estimated volume of spill (barrels, US) | Range: 0 to 904 bbl; Mean: 34.6 bbl; 1 barrel = approximately 159 litres |
| Qauntity_r | Float | Quantity of spill recovered (barrels) | Note: original column name misspelled in source data; Range: 0 to Estimated |
| Spill_stop | Date | Date containment was completed | 97 records missing; used to compute RTI |
| Type_of_fa | String | Type of facility where spill occurred | pl=pipeline (trunk or flow); fl=flowline (gathering line); mf=manifold; wh=wellhead |
| Cause | String | Cause of spill | sab=sabotage (pipeline vandalism and bunkering); cor=corrosion; eqf=equipment failure; other |
| Site_locat | String | Free-text description of incident location | Not standardised; for reference only |
| Latitude | Float | WGS84 latitude of incident location | Decimal degrees; Range: approximately 4.8N to 5.3N |
| Longitude | Float | WGS84 longitude of incident location | Decimal degrees; Range: approximately 6.2E to 6.8E |
| LGA | String | Local Government Area of incident | Ahoada-West (81.5%), Yenagoa (13.4%), Abua-Odual (4.2%), Ogba/Egbema/Ndoni (0.3%); some entries have leading/trailing whitespace |
| Estimate_1 | Float | Estimated affected area (hectares) | Derived from incident report; some zeroes where area not estimated |
| Spill_area | String | Type of surface affected | la=land; ss=swamp; sw=water surface; combinations: la,ss; ss,sw; iw=inland waterway |
| Descriptio | String | Narrative description of incident | Free text; JIV findings and site conditions |

---

## Derived Variables

The following variables are computed during preprocessing and used in ML models:

| Derived Variable | Formula | Description |
|-----------------|---------|-------------|
| response_days | Spill_stop minus Incident_d in days | Response Time Index (RTI); missing for 97 records |
| report_lag | Report_dat minus Incident_d in days | Days from incident to NOSDRA notification |
| CER | (Qauntity_r / Estimated) * 100 | Containment Efficiency Ratio (%); available for n=278 |
| log_volume | log(Estimated + 1) | Log-transformed spill volume |
| month | Incident_d.month | Calendar month of incident (1-12) |
| year | Incident_d.year | Calendar year (2016-2024) |
| month_sin | sin(2 * pi * month / 12) | Cyclic month encoding (sine component) |
| month_cos | cos(2 * pi * month / 12) | Cyclic month encoding (cosine component) |
| dry_season | 1 if month in {11,12,1,2}, else 0 | Binary dry season indicator |
| is_sabotage | 1 if Cause == 'sab', else 0 | Binary sabotage indicator |
| SVS | min-max normalised log_volume | Spill Volume Score for PHRI |
| RTS | min-max normalised response_days | Response Time Score for PHRI |
| CSS | {cr=1.0, co=0.67, no=0.33, other=0.0} | Contamination Severity Score for PHRI |
| EVS | {sw or ss=1.0, mixed=0.67, la=0.33} | Ecosystem Vulnerability Score for PHRI |
| FRS | {pl=1.0, fl=0.67, mf=0.33, wh=0.33} | Facility Risk Score for PHRI |
| PHRI | 0.30*SVS + 0.25*RTS + 0.20*CSS + 0.15*EVS + 0.10*FRS | Public Health Risk Index [0,1] |
| PHRI_class | Low if PHRI<0.33; Medium if 0.33-0.67; High if PHRI>=0.67 | Discretised risk class |
| CER_class | Low if CER<33%; Medium if 33-67%; High if CER>=67% | Discretised CER class |

---

## Data Quality Notes

1. **LGA standardisation:** The LGA column contains several variants of the same LGA name (e.g., "Ahoada-West", "AHOADA WEST", "Ahoada-West '"). All should be standardised to "Ahoada-West" during preprocessing.

2. **Contaminant encoding:** Several records have non-standard values in the Contaminan column ("NIL", "none", "other: Locally constructed boom", etc.). These are grouped into the "other" category during preprocessing.

3. **Missing Spill_stop dates:** 97 records (28.9%) have missing Spill_stop dates. These are excluded from RTI analysis (n=238 available) but retained for PHRI analysis with response_days imputed by facility-type-stratified median.

4. **Missing Report_dat:** 30 records (8.9%) have missing Report_dat values. These are imputed using the company-stratified median report lag.

5. **CER outliers:** 14 records produce CER values above 100% due to inconsistencies between estimated and recovered volumes in the source data. These are capped at 100% during preprocessing.

6. **Volume distribution:** The Estimated column is heavily right-skewed (max=904 bbl, mean=34.6 bbl, median=3.2 bbl). Log transformation is applied in all ML models.

7. **Coordinate precision:** Latitude and Longitude values are provided in decimal degrees to 5-6 decimal places. Some coordinates may represent the general pipeline corridor rather than the exact spill point.

---

## Encoding Reference

### Contaminan (Contaminant Type)
| Code | Description | CSS Score | Health Risk |
|------|-------------|-----------|-------------|
| cr | Crude oil | 1.00 | Highest (PAH, BTEX carcinogens) |
| co | Condensate (light hydrocarbons) | 0.67 | High (volatile, acute respiratory) |
| no | Produced water (formation brine) | 0.33 | Moderate (NORM, heavy metals) |
| other / nil / none | Unknown or other | 0.00 | Unclassified |

### Type_of_fa (Facility Type)
| Code | Description | FRS Score | Typical Diameter |
|------|-------------|-----------|-----------------|
| pl | Trunk pipeline or flow pipeline | 1.00 | 6-24 inches |
| fl | Flowline (well gathering line) | 0.67 | 2-4 inches |
| mf | Manifold | 0.33 | — |
| wh | Wellhead | 0.33 | — |

### Cause
| Code | Description | Proportion |
|------|-------------|------------|
| sab | Sabotage (pipeline vandalism, bunkering, hot tapping) | 97.3% |
| cor | Corrosion (material degradation) | 0.9% |
| eqf | Equipment failure (mechanical) | 0.6% |
| other / blank | Other or unspecified | 1.2% |

### Spill_area (Affected Surface Type)
| Code | Description | EVS Score |
|------|-------------|-----------|
| la | Land (dry or agricultural) | 0.33 |
| ss | Swamp (mangrove, wetland) | 1.00 |
| sw | Water surface (river, creek, sea) | 1.00 |
| la,ss | Mixed land and swamp | 0.67 |
| ss,sw | Mixed swamp and water | 1.00 |
| iw | Inland waterway | 0.67 |
| (blank) | Unknown | 0.17 |

---

## Suggested Preprocessing Code

```python
import pandas as pd
import numpy as np

def load_and_clean(filepath):
    df = pd.read_csv(filepath, encoding='latin1')

    # Parse dates
    for col in ['Incident_d', 'Report_dat', 'Spill_stop']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Standardise LGA
    df['LGA'] = df['LGA'].str.strip().str.title()
    lga_map = {"Ahoada-West '": 'Ahoada-West', 'Ahoada West': 'Ahoada-West'}
    df['LGA'] = df['LGA'].replace(lga_map)

    # Standardise contaminant
    df['Contaminan'] = df['Contaminan'].str.strip().str.lower()
    df['Contaminan_clean'] = df['Contaminan'].apply(
        lambda x: x if x in ['cr', 'co', 'no'] else 'other'
    )

    # Derived date features
    df['year'] = df['Incident_d'].dt.year
    df['month'] = df['Incident_d'].dt.month
    df['response_days'] = (df['Spill_stop'] - df['Incident_d']).dt.days.clip(lower=0)
    df['report_lag'] = (df['Report_dat'] - df['Incident_d']).dt.days.clip(lower=0)

    # Impute missing response_days by facility-type median
    med = df.groupby('Type_of_fa')['response_days'].median()
    df['response_days'] = df.apply(
        lambda r: med.get(r['Type_of_fa'], df['response_days'].median())
        if pd.isna(r['response_days']) else r['response_days'], axis=1
    )

    # CER
    df['CER'] = np.where(
        df['Estimated'] > 0,
        (df['Qauntity_r'] / df['Estimated']) * 100,
        np.nan
    ).clip(0, 100)

    return df
```

---

## Contact and Data Access

For data access requests, contact NOSDRA directly:  
National Oil Spill Detection and Response Agency  
Plot 1100, Cadastral Zone A00, Central Business District, Abuja, Nigeria  
Web: www.nosdra.gov.ng
