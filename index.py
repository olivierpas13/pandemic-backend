from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from ODESolver import ForwardEuler
from sir import SIR
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI(title="Pandemic Backend", description="Disease transmition model using ODEs")


@app.get("/", response_class=HTMLResponse, tags=["Infected and Susceptible"])
async def calculate_data():
    beta = lambda t: 0.0005 if t <=10 else 0.0001

    sir = SIR(0.1, beta, 1500, 1, 0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 60, 1001)
    u, t = solver.solve(time_steps)

    # plt.plot(t, u[:, 0], label="Susceptible")
    # plt.plot(t, u[:, 1], label="Infected")
    # plt.plot(t, u[:, 2], label="Recovered")
    # plt.legend()
    # plt.show()
    return JSONResponse({"time": t.tolist(),"population": u.tolist()})