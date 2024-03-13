from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from ODESolver import ForwardEuler
from sir import SIR
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI(title="Pandemic Backend", description="Disease transmition model using ODEs")


@app.get("/", response_class=HTMLResponse, tags=["Default"])
async def calculate_default_data():
    beta = lambda t: 0.0005 if t <=10 else 0.0001

    sir = SIR(0.01, beta, 800, 1, 0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 60, 60)
    u, t = solver.solve(time_steps)
    return JSONResponse({"time": t.tolist(),"population": u.tolist()})

@app.get("/withconstants", response_class=HTMLResponse, tags=["Different constants"])

async def calculate_data_with_different_constant_values(nu: float =0.01, beta: float= 0.0003):

    sir = SIR(nu, beta, 800, 1, 0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 60, 60)
    u, t = solver.solve(time_steps)
    return JSONResponse({"time": t.tolist(),"population": u.tolist()})

@app.get("/initialvalues", response_class=HTMLResponse, tags=["Different population"])
async def calculate_data_with_different_inital_values(susceptible: int= 800, infected: int = 1, recovered: int = 0):
    
    beta = lambda t: 0.0005 if t <=10 else 0.0001
    sir = SIR(0.01, beta, susceptible, infected, recovered)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 60, 60)
    u, t = solver.solve(time_steps)
    return JSONResponse({"time": t.tolist(),"population": u.tolist()})

@app.get("/withoutrecovered", response_class=HTMLResponse, tags=["Different constants"])
async def calculate_data_without_recovered():
    
    beta = lambda t: 0.0005 if t <=10 else 0.0001
    sir = SIR(0, beta, 800, 1, 0)
    solver = ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)

    time_steps = np.linspace(0, 60, 60)
    u, t = solver.solve(time_steps)
    return JSONResponse({"time": t.tolist(),"population": u.tolist()})
