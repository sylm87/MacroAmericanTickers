import csv
from pathlib import Path
from datetime import date, timedelta, datetime

import pandas as pd
import yfinance as yf


TICKERS = """
NASDAQ:COKE,NYSE:PG,NYSE:JNJ,NASDAQ:AAPL,NASDAQ:MSFT,NYSE:DIS,NASDAQ:GOOGL,
NYSE:CVX,NYSE:XOM,NYSE:NKE,NYSE:STEM,NASDAQ:ONDS,NASDAQ:PLTR,NASDAQ:TSLA,
NASDAQ:META,NASDAQ:NFLX,NASDAQ:AMZN,NASDAQ:AMD,NYSE:ORCL,NASDAQ:AVGO,
NYSE:BABA,NYSE:TSM
""".replace("\n", "")

TICKER_DOLLARIDX = "TVC:DXY"
HISTORY_DIR = Path("rsi_history_tests")
LOOKBACK_DAYS = 15
RSI_PERIOD = 14
YF_PERIOD = "6mo"
YF_INTERVAL = "1d"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def tradingview_to_yahoo(tv_code: str) -> str:
    exchange, symbol = tv_code.split(":", 1)
    exchange = exchange.upper()
    symbol = symbol.upper()

    if exchange in {"NASDAQ", "NYSE", "AMEX"}:
        return symbol

    if exchange == "BME":
        return f"{symbol}.MC"

    if exchange == "TVC" and symbol == "DXY":
        return "DX-Y.NYB"

    if exchange == "CAPITALCOM" and symbol == "DXY":
        return "DX-Y.NYB"

    raise ValueError(f"No hay mapeo Yahoo definido para {tv_code}")


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_price_history(
    yahoo_ticker: str,
    period: str = YF_PERIOD,
    interval: str = YF_INTERVAL
) -> pd.DataFrame:
    df = yf.download(
        yahoo_ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise ValueError(f"Sin datos para {yahoo_ticker}")

    df = df.copy()

    # Protección extra por si alguna versión devolviera MultiIndex igualmente
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append(col[0])
            else:
                flat_cols.append(col)
        df.columns = flat_cols

    if "Close" not in df.columns:
        raise ValueError(
            f"No existe columna Close para {yahoo_ticker}. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"Sin cierres válidos para {yahoo_ticker}")

    return df


def get_rsi_values(
    tv_code: str,
    lookback_days: int = LOOKBACK_DAYS,
    rsi_period: int = RSI_PERIOD
) -> dict:
    yahoo_ticker = tradingview_to_yahoo(tv_code)
    df = fetch_price_history(yahoo_ticker)

    df["RSI"] = rsi_wilder(df["Close"], rsi_period)
    rsi_series = df["RSI"].dropna()

    if len(rsi_series) < 2:
        raise ValueError(f"No hay suficientes datos RSI para {tv_code} -> {yahoo_ticker}")

    last_n = rsi_series.tail(lookback_days)

    return {
        "ticker": tv_code,
        "yahoo_ticker": yahoo_ticker,
        "rsi_0": float(rsi_series.iloc[-1]),
        "rsi_1": float(rsi_series.iloc[-2]),
        "rsi_last_n": [float(x) for x in last_n.tolist()],
        "rsi_last_n_dates": [idx.strftime("%Y/%m/%d") for idx in last_n.index],
    }


def has_rsi_below_30(row: dict) -> bool:
    rsi_0 = row.get("rsi_0")
    rsi_1 = row.get("rsi_1")
    return (
        (rsi_0 is not None and rsi_0 < 30) or
        (rsi_1 is not None and rsi_1 < 30)
    )


def format_rsi(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def ticker_to_filename(tv_code: str) -> str:
    return tv_code.replace(":", "_") + ".csv"


def get_yesterday_str() -> str:
    yesterday = date.today() - timedelta(days=1)
    return yesterday.strftime("%Y/%m/%d")


def ensure_history_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def parse_date_safe(value: str):
    try:
        return datetime.strptime(value, "%Y/%m/%d").date()
    except Exception:
        return None


def load_csv_rows(csv_path: Path):
    rows = []
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return rows

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {
                "date": (row.get("date") or "").strip(),
                "ticker": (row.get("ticker") or "").strip(),
                "yahoo_ticker": (row.get("yahoo_ticker") or "").strip(),
                "rsi": (row.get("rsi") or "").strip(),
                "dxy_alignment": (row.get("dxy_alignment") or "").strip(),
            }
            if normalized["date"]:
                rows.append(normalized)

    return rows


def dedupe_and_sort_rows(rows: list):
    by_date = {}
    duplicates_removed = 0

    for row in rows:
        row_date = row["date"]
        if row_date in by_date:
            duplicates_removed += 1
        by_date[row_date] = row

    sorted_rows = sorted(
        by_date.values(),
        key=lambda r: (
            parse_date_safe(r["date"]) is None,
            parse_date_safe(r["date"]) or date.min
        )
    )

    return sorted_rows, duplicates_removed


def rewrite_csv(csv_path: Path, rows: list):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "ticker", "yahoo_ticker", "rsi", "dxy_alignment"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_historical_rsi_rows(
    tv_code: str,
    min_rows: int = LOOKBACK_DAYS,
    rsi_period: int = RSI_PERIOD
) -> list:
    yahoo_ticker = tradingview_to_yahoo(tv_code)
    df = fetch_price_history(yahoo_ticker)

    df["RSI"] = rsi_wilder(df["Close"], rsi_period)
    rsi_series = df["RSI"].dropna()

    if len(rsi_series) < min_rows:
        raise ValueError(
            f"No hay suficiente histórico RSI para {tv_code} -> {yahoo_ticker}. "
            f"Disponibles: {len(rsi_series)}, requeridos: {min_rows}"
        )

    hist = rsi_series.tail(min_rows)

    rows = []
    for idx, value in hist.items():
        rows.append({
            "date": idx.strftime("%Y/%m/%d"),
            "ticker": tv_code,
            "yahoo_ticker": yahoo_ticker,
            "rsi": f"{float(value):.6f}",
        })

    return rows


def ensure_minimum_history_rows(
    tv_code: str,
    min_rows: int = LOOKBACK_DAYS,
    dxy_alignment_default: str = "0"
):
    ensure_history_dir()

    csv_path = HISTORY_DIR / ticker_to_filename(tv_code)
    existing_rows = load_csv_rows(csv_path)
    existing_rows, duplicates_removed = dedupe_and_sort_rows(existing_rows)

    existing_dates = {row["date"] for row in existing_rows}

    if len(existing_rows) >= min_rows:
        if duplicates_removed > 0:
            rewrite_csv(csv_path, existing_rows)
            return False, f"histórico suficiente; reparados {duplicates_removed} duplicados"
        return False, f"histórico suficiente ({len(existing_rows)} registros)"

    historical_rows = get_historical_rsi_rows(tv_code, min_rows=min_rows)

    added = 0
    for hist_row in historical_rows:
        if hist_row["date"] not in existing_dates:
            existing_rows.append({
                "date": hist_row["date"],
                "ticker": hist_row["ticker"],
                "yahoo_ticker": hist_row["yahoo_ticker"],
                "rsi": hist_row["rsi"],
                "dxy_alignment": dxy_alignment_default,
            })
            existing_dates.add(hist_row["date"])
            added += 1

    existing_rows, duplicates_removed_after = dedupe_and_sort_rows(existing_rows)
    rewrite_csv(csv_path, existing_rows)

    total_rows = len(existing_rows)
    total_dup = duplicates_removed + duplicates_removed_after

    if total_dup > 0:
        return True, f"rellenado histórico (+{added}); total={total_rows}; reparados {total_dup} duplicados"

    return True, f"rellenado histórico (+{added}); total={total_rows}"


def save_yesterday_rsi_if_missing(
    tv_code: str,
    yahoo_ticker: str,
    rsi_yesterday,
    dxy_alignment: bool
):
    if rsi_yesterday is None:
        return False, "RSI[1] vacío, no guardado"

    ensure_history_dir()

    csv_path = HISTORY_DIR / ticker_to_filename(tv_code)
    target_date = get_yesterday_str()

    existing_rows = load_csv_rows(csv_path)
    normalized_rows, duplicates_removed = dedupe_and_sort_rows(existing_rows)
    existing_dates = {row["date"] for row in normalized_rows}

    if target_date in existing_dates:
        rewrite_csv(csv_path, normalized_rows)
        if duplicates_removed > 0:
            return False, f"ya existía {target_date}; reparados {duplicates_removed} duplicados"
        return False, f"ya existía registro para {target_date}"

    new_row = {
        "date": target_date,
        "ticker": tv_code,
        "yahoo_ticker": yahoo_ticker,
        "rsi": f"{float(rsi_yesterday):.6f}",
        "dxy_alignment": "1" if dxy_alignment else "0",
    }

    normalized_rows.append(new_row)
    normalized_rows, duplicates_removed_after = dedupe_and_sort_rows(normalized_rows)
    total_duplicates_removed = duplicates_removed + duplicates_removed_after

    rewrite_csv(csv_path, normalized_rows)

    if total_duplicates_removed > 0:
        return True, f"guardado {target_date}; reparados {total_duplicates_removed} duplicados"

    return True, f"guardado {target_date}"


def get_last_n_rows_for_ticker(tv_code: str, n: int) -> list:
    csv_path = HISTORY_DIR / ticker_to_filename(tv_code)
    rows = load_csv_rows(csv_path)
    rows, _ = dedupe_and_sort_rows(rows)
    return rows[-n:]


def any_rsi_below_30_in_last_n_rows(tv_code: str, n: int):
    rows = get_last_n_rows_for_ticker(tv_code, n)

    matching_rows = []
    for row in rows:
        try:
            rsi_value = float(row["rsi"])
            if rsi_value < 30:
                matching_rows.append(row)
        except Exception:
            continue

    return len(matching_rows) > 0, matching_rows


def print_section(title: str):
    print("=" * 100)
    print(title)
    print("=" * 100)


def main():
    tickers = [x.strip() for x in TICKERS.split(",") if x.strip()]
    results = []
    historical_candidates = []

    try:
        dollar_data = get_rsi_values(TICKER_DOLLARIDX)
        dollar_oversold = has_rsi_below_30(dollar_data)

        _, history_msg = ensure_minimum_history_rows(
            dollar_data["ticker"],
            min_rows=LOOKBACK_DAYS
        )

        _, msg_save = save_yesterday_rsi_if_missing(
            tv_code=dollar_data["ticker"],
            yahoo_ticker=dollar_data["yahoo_ticker"],
            rsi_yesterday=dollar_data.get("rsi_1"),
            dxy_alignment=dollar_oversold
        )
        dollar_data["csv_status"] = f"{history_msg} | {msg_save}"
    except Exception as e:
        dollar_data = {
            "ticker": TICKER_DOLLARIDX,
            "error": str(e)
        }
        dollar_oversold = False

    print_section("RSI DEL ÍNDICE DÓLAR")

    if "error" in dollar_data:
        print(f"{RED}{dollar_data['ticker']} | ERROR: {dollar_data['error']}{RESET}")
    else:
        dollar_line = (
            f"{dollar_data['ticker']} ({dollar_data['yahoo_ticker']}) | "
            f"RSI={format_rsi(dollar_data['rsi_0'])} | "
            f"RSI[1]={format_rsi(dollar_data['rsi_1'])} | "
            f"CSV: {dollar_data.get('csv_status', '-')}"
        )

        if dollar_oversold:
            print(f"{GREEN}{dollar_line} | DÓLAR CON RSI < 30{RESET}")
        else:
            print(f"{CYAN}{dollar_line}{RESET}")

    print()
    print_section("RSI DE ACCIONES")

    for tv_code in tickers:
        try:
            row = get_rsi_values(tv_code)
            stock_oversold = has_rsi_below_30(row)
            alignment_detected = stock_oversold and dollar_oversold

            _, history_msg = ensure_minimum_history_rows(
                row["ticker"],
                min_rows=LOOKBACK_DAYS
            )

            _, msg_save = save_yesterday_rsi_if_missing(
                tv_code=row["ticker"],
                yahoo_ticker=row["yahoo_ticker"],
                rsi_yesterday=row.get("rsi_1"),
                dxy_alignment=alignment_detected
            )

            row["csv_status"] = f"{history_msg} | {msg_save}"
            row["alignment_detected"] = alignment_detected
            results.append(row)

        except Exception as e:
            results.append({
                "ticker": tv_code,
                "error": str(e)
            })

    for row in results:
        if "error" in row:
            print(f"{RED}{row['ticker']} | ERROR: {row['error']}{RESET}")
            continue

        stock_oversold = has_rsi_below_30(row)

        base_line = (
            f"{row['ticker']} ({row['yahoo_ticker']}) | "
            f"RSI={format_rsi(row['rsi_0'])} | "
            f"RSI[1]={format_rsi(row['rsi_1'])} | "
            f"CSV: {row.get('csv_status', '-')}"
        )

        if stock_oversold:
            print(f"{GREEN}{base_line} | RSI < 30 en la acción{RESET}")
        else:
            print(base_line)

        if row.get("alignment_detected"):
            print(
                f"{YELLOW}  >>> ALINEACIÓN DETECTADA: "
                f"{row['ticker']} y {dollar_data['ticker']} tienen RSI < 30{RESET}"
            )

    print()
    print_section(f"RESUMEN HISTÓRICO ÚLTIMOS {LOOKBACK_DAYS} CIERRES")

    if "error" in dollar_data:
        print(f"{RED}No se pudo evaluar el resumen histórico porque DXY falló en esta ejecución.{RESET}")
    else:
        dxy_had_oversold, dxy_matching_rows = any_rsi_below_30_in_last_n_rows(
            dollar_data["ticker"],
            LOOKBACK_DAYS
        )

        if not dxy_had_oversold:
            print(
                f"{CYAN}DXY no ha tenido RSI < 30 en sus últimos {LOOKBACK_DAYS} cierres guardados. "
                f"No hay coincidencias potenciales que mostrar.{RESET}"
            )
        else:
            for row in results:
                if "error" in row:
                    continue

                asset_had_oversold, asset_matching_rows = any_rsi_below_30_in_last_n_rows(
                    row["ticker"],
                    LOOKBACK_DAYS
                )

                if asset_had_oversold:
                    historical_candidates.append({
                        "ticker": row["ticker"],
                        "asset_matches": asset_matching_rows,
                    })

            if not historical_candidates:
                print(
                    f"{CYAN}DXY sí tuvo RSI < 30 en los últimos {LOOKBACK_DAYS} cierres, "
                    f"pero ningún activo de la lista lo tuvo en ese mismo tramo histórico.{RESET}"
                )
            else:
                print(
                    f"{MAGENTA}Activos potencialmente coincidentes con DXY "
                    f"en los últimos {LOOKBACK_DAYS} cierres guardados:{RESET}"
                )

                for candidate in historical_candidates:
                    dates_str = ", ".join(match["date"] for match in candidate["asset_matches"])
                    print(
                        f"{YELLOW}- {candidate['ticker']} | "
                        f"fechas del activo con RSI < 30: {dates_str}{RESET}"
                    )

                dxy_dates = ", ".join(match["date"] for match in dxy_matching_rows)
                print()
                print(f"{MAGENTA}Fechas de DXY con RSI < 30 en ese mismo tramo: {dxy_dates}{RESET}")

    print()
    print_section("FIN")


if __name__ == "__main__":
    main()