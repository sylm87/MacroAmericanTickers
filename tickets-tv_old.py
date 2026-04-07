from tradingview_ta import TA_Handler, Interval
from pathlib import Path
from datetime import date, timedelta, datetime
import csv
import time


TICKERS = """
NASDAQ:COKE,NYSE:PG,NYSE:JNJ,NASDAQ:AAPL,NASDAQ:MSFT,NYSE:DIS,NASDAQ:GOOGL,
NYSE:CVX,NYSE:XOM,NYSE:NKE,NYSE:STEM,NASDAQ:ONDS,NASDAQ:PLTR,NASDAQ:TSLA,
NASDAQ:META,NASDAQ:NFLX,NASDAQ:AMZN,NASDAQ:AMD,NYSE:ORCL,NASDAQ:AVGO,
NYSE:BABA,NYSE:TSM,BME:IDR
""".replace("\n", "")

TICKER_DOLLARIDX = "TVC:DXY"
HISTORY_DIR = Path("rsi_history")
LOOKBACK_DAYS = 15

# Control de rate limit / reintentos
RATE_LIMIT_SLEEP_SECONDS = 300   # 5 minutos
MAX_RETRIES_429 = 3

# Pausa entre peticiones normales para reducir riesgo de 429
REQUEST_DELAY_SECONDS = 3

# Colores ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def resolve_screener(exchange: str, symbol: str) -> str:
    exchange = exchange.upper()
    symbol = symbol.upper()

    if exchange in {"BINANCE", "BYBIT", "COINBASE", "KRAKEN", "BITSTAMP", "CRYPTOCAP"}:
        return "crypto"

    if exchange in {"CAPITALCOM", "OANDA", "FX_IDC", "FOREXCOM", "PEPPERSTONE"}:
        return "forex"

    if exchange in {"BME"}:
        return "spain"

    if exchange in {"NASDAQ", "NYSE", "AMEX"}:
        return "america"

    if exchange in {"TVC"}:
        return "america"

    return "america"


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text or
        "too many requests" in text or
        "rate limit" in text
    )


def sleep_between_requests():
    time.sleep(REQUEST_DELAY_SECONDS)


def fetch_indicators_with_retry(
    symbol: str,
    exchange: str,
    screener: str,
    interval,
    indicators,
    timeout: int = 10
):
    last_error = None

    for attempt in range(MAX_RETRIES_429 + 1):
        try:
            handler = TA_Handler(
                symbol=symbol,
                exchange=exchange,
                screener=screener,
                interval=interval,
                timeout=timeout
            )
            data = handler.get_indicators(indicators)

            # Pausa tras petición exitosa
            sleep_between_requests()
            return data

        except Exception as e:
            last_error = e

            if is_rate_limit_error(e):
                if attempt < MAX_RETRIES_429:
                    wait_seconds = RATE_LIMIT_SLEEP_SECONDS
                    print(
                        f"{YELLOW}[RATE LIMIT] 429 detectado para {exchange}:{symbol}. "
                        f"Esperando {wait_seconds} segundos antes del reintento "
                        f"({attempt + 1}/{MAX_RETRIES_429})...{RESET}"
                    )
                    time.sleep(wait_seconds)
                    continue

            raise

    raise last_error


def get_rsi_values(tv_code: str, interval=Interval.INTERVAL_1_DAY) -> dict:
    exchange, symbol = tv_code.split(":", 1)
    screener = resolve_screener(exchange, symbol)

    data = fetch_indicators_with_retry(
        symbol=symbol,
        exchange=exchange,
        screener=screener,
        interval=interval,
        indicators=["RSI", "RSI[1]"],
        timeout=10
    )

    return {
        "ticker": tv_code,
        "screener": screener,
        "rsi_0": data.get("RSI"),
        "rsi_1": data.get("RSI[1]"),
    }


def get_dollar_index_rsi(interval=Interval.INTERVAL_1_DAY) -> dict:
    candidates = [
        ("TVC:DXY", "america"),
        ("CAPITALCOM:DXY", "forex"),
    ]

    errors = []

    for idx, (tv_code, screener) in enumerate(candidates):
        exchange, symbol = tv_code.split(":", 1)
        try:
            data = fetch_indicators_with_retry(
                symbol=symbol,
                exchange=exchange,
                screener=screener,
                interval=interval,
                indicators=["RSI", "RSI[1]"],
                timeout=10
            )
            return {
                "ticker": tv_code,
                "screener": screener,
                "rsi_0": data.get("RSI"),
                "rsi_1": data.get("RSI[1]"),
            }
        except Exception as e:
            errors.append(f"{tv_code} [{screener}] -> {e}")

            # Pequeña pausa también entre candidatos alternativos
            if idx < len(candidates) - 1:
                sleep_between_requests()

    raise Exception(" ; ".join(errors))


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
                "screener": (row.get("screener") or "").strip(),
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
        key=lambda r: (parse_date_safe(r["date"]) is None, parse_date_safe(r["date"]) or date.min)
    )

    return sorted_rows, duplicates_removed


def rewrite_csv(csv_path: Path, rows: list):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "ticker", "screener", "rsi", "dxy_alignment"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_yesterday_rsi_if_missing(
    tv_code: str,
    screener: str,
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
        "screener": screener,
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
        dollar_data = get_dollar_index_rsi()
        dollar_oversold = has_rsi_below_30(dollar_data)

        _, msg = save_yesterday_rsi_if_missing(
            tv_code=dollar_data["ticker"],
            screener=dollar_data["screener"],
            rsi_yesterday=dollar_data.get("rsi_1"),
            dxy_alignment=dollar_oversold
        )
        dollar_data["csv_status"] = msg
    except Exception as e:
        dollar_data = {
            "ticker": "DXY",
            "error": str(e)
        }
        dollar_oversold = False

    print_section("RSI DEL ÍNDICE DÓLAR")

    if "error" in dollar_data:
        print(f"{RED}{dollar_data['ticker']} | ERROR: {dollar_data['error']}{RESET}")
    else:
        dollar_line = (
            f"{dollar_data['ticker']} | "
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

            _, msg = save_yesterday_rsi_if_missing(
                tv_code=row["ticker"],
                screener=row["screener"],
                rsi_yesterday=row.get("rsi_1"),
                dxy_alignment=alignment_detected
            )

            row["csv_status"] = msg
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
            f"{row['ticker']} | "
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