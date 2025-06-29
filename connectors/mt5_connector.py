import MetaTrader5 as mt5
from datetime import datetime
from config import MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOLS_MAPPING
from utils import log

_initialized = False

def initialize_mt5():
    global _initialized
    if _initialized:
        return
    # start the terminal if not already running
    if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log(f"[MT5] Initialization failed: {mt5.last_error()}")
        raise RuntimeError("MT5 init failed")
    log("[MT5] Initialized and connected successfully")
    _initialized = True

def shutdown_mt5():
    global _initialized
    if _initialized:
        mt5.shutdown()
        log("[MT5] Shutdown complete")
        _initialized = False

def mt5_symbol(symbol: str) -> str:
    # Map human-readable pair to broker symbol
    return SYMBOLS_MAPPING.get(symbol, symbol.replace("/", ""))

def get_current_price(symbol: str) -> float:
    initialize_mt5()
    sym = mt5_symbol(symbol)
    tick = mt5.symbol_info_tick(sym)
    if not tick:
        log(f"[MT5] Failed to get tick for {sym}")
        raise RuntimeError(f"No tick for {sym}")
    price = (tick.ask + tick.bid) / 2
    return price

def place_order(symbol: str, action: int, volume: float, sl: float = None, tp: float = None):
    """
    action: 1 = buy, -1 = sell
    volume: lots
    sl, tp: absolute price levels (optional)
    """
    initialize_mt5()
    sym = mt5_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        log(f"[MT5] Failed to select symbol {sym}")
        return None

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(sym).ask if action == 1 else mt5.symbol_info_tick(sym).bid,
        "deviation": 10,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if sl:
        req["sl"] = sl
    if tp:
        req["tp"] = tp

    result = mt5.order_send(req)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"[MT5] Order send failed ({sym}): {result.comment}")
        return None
    log(f"[MT5] Order placed: {sym} {'BUY' if action==1 else 'SELL'} vol={volume}")
    return result

def get_open_positions(symbol: str):
    initialize_mt5()
    sym = mt5_symbol(symbol)
    positions = mt5.positions_get(symbol=sym)
    return positions or []

def close_position(position):
    initialize_mt5()
    sym = position.symbol
    action = -1 if position.type == mt5.POSITION_TYPE_BUY else 1
    volume = position.volume
    price = mt5.symbol_info_tick(sym).bid if action == 1 else mt5.symbol_info_tick(sym).ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if action == -1 else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "position": position.ticket,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"[MT5] Close failed ({sym}): {result.comment}")
        return None
    log(f"[MT5] Position closed: {sym} ticket={position.ticket}")
    return result
