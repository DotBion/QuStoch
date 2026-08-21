import os

from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from stockData import get_stock_info
from quantumMonteCarloStochastic import Q_sim_start

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins


def parse_money(raw):
    """Parse a user-entered price like ``"$1,234.50"`` into a float."""
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    value = float(cleaned)
    if value <= 0:
        raise ValueError("price must be positive")
    return value


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on("search_event")
def handle_search_event(data):
    search_text = (data or {}).get("search_text", "").strip()
    print(f"Received ticker: {search_text}")

    try:
        result = get_stock_info(search_text)
        result.update({"status": "success"})
        emit("search_response", result)

    except Exception as exc:
        print(f"lookup failed for {search_text!r}: {exc}")
        emit("search_response", {"status": "error", "message": "Invalid ticker, try again."})


@socketio.on("backend_simulation_event")
def handle_simulation_submission(data):
    data = data or {}
    print(data)

    try:
        strike = parse_money(data["striking_price"])
        maturity_weeks = int(data["maturity_time"])
        if maturity_weeks <= 0:
            raise ValueError("maturity must be at least one week")
    except (KeyError, TypeError, ValueError) as exc:
        print(f"invalid simulation inputs: {exc}")
        emit(
            "backend_simulation_event",
            {"status": "error", "message": "Please enter a positive strike price and maturity."},
        )
        return

    try:
        stock_data = get_stock_info(data["stock_ticker"])
    except Exception as exc:
        print(f"market data unavailable: {exc}")
        emit(
            "backend_simulation_event",
            {"status": "error", "message": "Could not load market data for that ticker."},
        )
        return

    print(stock_data)

    try:
        graph_path, detailed_output = Q_sim_start(
            round(stock_data["latest_price"], 2),
            round(strike, 2),
            maturity_weeks,
            round(stock_data["risk_free_return"], 4),
            round(stock_data["historical_volatility"], 4),
        )
    except Exception as exc:
        print(f"simulation failed: {exc}")
        emit(
            "backend_simulation_event",
            {"status": "error", "message": "The simulation could not be completed."},
        )
        return

    print("sending....")

    emit(
        "backend_simulation_event",
        {
            "graph_path": graph_path,
            "detailed_output": detailed_output,
            "status": "success",
        },
    )


if __name__ == '__main__':
    socketio.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
