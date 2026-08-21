# QuStoch: Quantum-Enhanced Stochastic Market Simulations

## Overview

Traditional stock market simulations rely on classical stochastic models like Monte Carlo methods, which, while effective, can be computationally intensive and limited in capturing the full range of market dynamics.

**QuStoch** leverages the power of **quantum superposition** to simulate multiple stochastic market states simultaneously. By integrating quantum algorithms into Brownian motion simulations, QuStoch offers a more efficient and probabilistically rich approach to financial modeling.

---

## Features

- **Quantum-Powered Simulations**  
  Harnesses quantum superposition to evaluate multiple stock price trajectories in parallel.

- **Enhanced Brownian Motion Modeling**  
  Encodes stochastic paths into qubits for a more representative market simulation.

- **Interactive Visualization**  
  Web-based dashboard that displays simulation results in real time.

---

## How It Works

Instead of iterating through all possible stock price movements like classical Monte Carlo simulations, **QuStoch** leverages quantum computing to represent many market states concurrently:

1. **Quantum Computation**  
   A quantum circuit encodes stochastic paths into qubits using **Hadamard** and **controlled rotation gates**.

2. **Classical Processing**  
   Quantum-generated results are processed using Python and calibrated against historical stock data.

3. **Web-Based Visualization**  
   The output is rendered dynamically in a Flask-based web dashboard with **JavaScript** and **Matplotlib**.

---

## Technologies Used

- **Quantum Frameworks** – Quantum circuits for stochastic modeling.
- **Python Backend** – Data processing and statistical analysis.
- **Web Frontend** – Built with Flask, HTML/CSS, and JavaScript.
- **Visualization** – Matplotlib for graphical stock behavior representation.

---

## Challenges and Solutions

- **Efficient Quantum Simulation**  
  Designed a practical quantum algorithm to map probability amplitudes effectively.

- **Validating the Quantum Result**  
  Runs on a noiseless state-vector simulator, and checks the amplitude-estimation output
  against the classical expectation over the same discretised distribution.

- **Hybrid Integration**  
  Combined quantum output with classical analytics for actionable insights.

---

## Achievements

- Built a working **quantum-classical hybrid prototype** for financial modeling.
- Demonstrated **parallelized stochastic simulations** using quantum circuits.
- Developed a real-time, interactive web interface.
- Validated the simulation against closed-form **Black-Scholes** prices and Greeks.

---

## Key Learnings

- Applications of quantum computing in **probabilistic financial modeling**.
- Challenges in **quantum circuit design** for stochastic processes.
- Importance of hybrid quantum-classical workflows in real-world systems.

---

## Future Plans

- **Enhanced Quantum Algorithms**  
  Add noise models and error mitigation, and improve the state-preparation encoding so the
  circuit can target real NISQ hardware rather than a simulator.

- **Alternative Simulation Models**  
  Explore **quantum walks** and other novel models for financial forecasting.

- **Expanded Web Features**  
  Add **live stock tracking**, more granular analytics, and deeper visual insights.

---

## Get Started

Interested in experimenting with quantum-enhanced market simulations?

```bash
git clone https://github.com/DotBion/QuStoch.git
cd QuStoch

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/app.py
```

Then open <http://127.0.0.1:5000>, look up a ticker, enter a strike price and a maturity
in weeks, and run the simulation.

Environment variables: `HOST`, `PORT`, and `FLASK_DEBUG=1` to enable the Flask debugger
(leave it off unless you are developing locally).

To run the simulation on its own, without the web server:

```bash
python src/quantumMonteCarloStochastic.py
```

### Tests

```bash
pip install -r requirements.txt
pytest
```

The suite checks the simulation against closed-form Black-Scholes values and asserts that
the quantum amplitude estimate agrees with the classical expectation over the same
discretised distribution.
