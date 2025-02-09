from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
db = SQLAlchemy(app)

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(10), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)

@app.route('/')
def index():
    stocks = Stock.query.all()
    return render_template('index.html', stocks=stocks)

@app.route('/buy', methods=['POST'])
def buy():
    """
    Expects form data:
      - symbol
      - shares
      - price
    """
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    price = float(request.form['price'])  # Use the price from the client

    if shares > 0:
        # Either update existing stock or create a new one
        existing_stock = Stock.query.filter_by(name=symbol).first()
        if existing_stock:
            existing_stock.shares += shares
            # You could optionally recalculate price or set a new price here
            existing_stock.price = price
        else:
            stock = Stock(name=symbol, shares=shares, price=price)
            db.session.add(stock)
        db.session.commit()
    return redirect(url_for('index'))

@app.route('/sell', methods=['POST'])
def sell():
    """
    Expects form data:
      - symbol
      - shares
      - price
    """
    symbol = request.form['symbol'].upper()
    shares_to_sell = int(request.form['shares'])
    # We don't necessarily need the price to process the sell, 
    # but it's here if you want to use it.
    price = float(request.form['price'])  

    stock = Stock.query.filter_by(name=symbol).first()
    if stock and stock.shares >= shares_to_sell:
        stock.shares -= shares_to_sell
        if stock.shares == 0:
            db.session.delete(stock)
        db.session.commit()
    return redirect(url_for('index'))

# Example socket endpoints
@socketio.on("search_event")
def handle_search_event(data):
    search_text = data.get("search_text")
    print(f"Received ticker: {search_text}")
    # Dummy result – you can replace this logic to fetch from yfinance, etc.
    result = {
        "latest_price": 123.45,
        "latest_volume": 67890,
        "historical_volatility": 0.05,
        "status": "success"
    }
    emit("search_response", result)

@socketio.on("backend_simulaton_event")
def handle_simulation_submission(data):
    data = {
        "image_path": None,
        "detailed_output": {"example_key": "example_value"},
        "status": "success"
    }
    emit("backend_simulation_event", data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    # Or: socketio.run(app, debug=True)
