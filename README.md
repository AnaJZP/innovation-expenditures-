# Do intangible innovation expenditures raise value added? Evidence from Mexican industries, 2016–2024

Replication package: data, code and results for the article of the same title.

The study links the innovation module of the 2019 Mexican Economic Census (spending on R&D, software and databases, and patents and other industrial property in 2016–2018) with the KLEMS productivity accounts (value added, capital and hours, 1990–2024) and the 2018 input–output table. The panel covers 64 industries.

## Contents

```
analisis/
  02_construir_panel.py   builds the panels from the INEGI source files
  03_estimaciones.py      production functions, local projections, input–output measures, robustness
  04_figuras.py           figures (PNG 600 dpi, SVG, PDF)
datos_fuente/             INEGI source files read by the scripts, with their metadata
resultados/               datasets, tables and figures produced by the scripts
  figuras/                figures 1–3 of the article
  tablas_resultados.xlsx  all estimation tables
```

## Data sources

All data are public, aggregated statistics published by the Instituto Nacional de Estadística y Geografía (INEGI) and redistributed here under the [INEGI terms of free use](https://www.inegi.org.mx/inegi/terminos.html).

| Source | Files | Coverage | Release used |
|---|---|---|---|
| Censos Económicos 2019, innovation module, tables INNONCE19_07 and INNONCE19_01 | `datos_fuente/innonce19_07.xlsx`, `innonce19_01.xlsx` | Large economic units, 2016–2018, 86 SCIAN subsectors | 2020 |
| Productividad Total de los Factores, modelo KLEMS, base 2018 | `datos_fuente/ptf/` | 64 activities, 1990–2024 | Revised figures, metadata modified 2026-08-24 |
| Producto Interno Bruto Trimestral, serie detallada, base 2018 | `datos_fuente/pibt/` | Value added by subsector, used for checks | Q2 2026, metadata modified 2026-08-20 |
| Matriz de Insumo Producto 2018, industry by industry, domestic flows at basic prices | `datos_fuente/mip/` | Subsector level | Metadata modified 2023-08-25 |

Note on data vintages (checked November 2026):

- The KLEMS files are the latest release, with revised figures for 2024. INEGI's 2026 release calendar schedules preliminary figures for 2025 for 23 December 2026.
- Spending on innovation is available only for 2016–2018. The 2024 Economic Census did not repeat the 2019 module on spending on R&D, software and industrial property; it asks instead about the use of digital technologies.

## Reproducing the results

Python 3.12. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analisis/02_construir_panel.py
python analisis/03_estimaciones.py
python analisis/04_figuras.py
```

Each script reads from `datos_fuente/` or `resultados/` and writes to `resultados/`. The estimation log is saved in `resultados/log_estimaciones.txt`.
