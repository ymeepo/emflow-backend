from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json

app = FastAPI(title="State Storage API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state storage
state_store: Dict[str, Any] = {}

# In-memory funnel data storage
funnel_data: Dict[str, int] = {}

class StateItem(BaseModel):
    key: str
    value: Any

class StateValue(BaseModel):
    value: Any

class FunnelData(BaseModel):
    applicants: int
    hm_review: int
    hm_interview: int
    tech_screen: int
    panel_1: int
    panel_2: int
    hired: int

@app.get("/")
async def root():
    return {"message": "State Storage API"}

@app.get("/state")
async def get_all_state():
    return state_store

@app.get("/state/{key}")
async def get_state(key: str):
    if key not in state_store:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"key": key, "value": state_store[key]}

@app.post("/state")
async def set_state(item: StateItem):
    state_store[item.key] = item.value
    return {"message": "State updated", "key": item.key, "value": item.value}

@app.put("/state/{key}")
async def update_state(key: str, value: StateValue):
    state_store[key] = value.value
    return {"message": "State updated", "key": key, "value": value.value}

@app.delete("/state/{key}")
async def delete_state(key: str):
    if key not in state_store:
        raise HTTPException(status_code=404, detail="Key not found")
    deleted_value = state_store.pop(key)
    return {"message": "State deleted", "key": key, "value": deleted_value}

@app.delete("/state")
async def clear_all_state():
    state_store.clear()
    return {"message": "All state cleared"}

# Funnel endpoints
@app.get("/funnel")
async def get_funnel_data():
    if not funnel_data:
        raise HTTPException(status_code=404, detail="No funnel data found")
    return funnel_data

@app.post("/funnel")
async def save_funnel_data(data: FunnelData):
    funnel_data.update({
        "applicants": data.applicants,
        "hm_review": data.hm_review,
        "hm_interview": data.hm_interview,
        "tech_screen": data.tech_screen,
        "panel_1": data.panel_1,
        "panel_2": data.panel_2,
        "hired": data.hired
    })
    return {"message": "Funnel data saved", "data": funnel_data}

@app.delete("/funnel")
async def clear_funnel_data():
    funnel_data.clear()
    return {"message": "Funnel data cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)