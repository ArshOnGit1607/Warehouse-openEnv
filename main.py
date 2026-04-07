from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
import math

class Item(BaseModel):
    id: str
    size: float

class Box(BaseModel):
    id: str
    capacity: float
    used_space: float

class Observation(BaseModel):
    remaining_items: List[Item]
    boxes: List[Box]

class Action(BaseModel):
    action_type: Literal["place_item", "open_new_box"]
    item_id: Optional[str] = None
    box_id: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class EnvState:
    def __init__(self):
        self.remaining_items: List[Item] = []
        self.boxes: List[Box] = []
        self.task_name: str = "task_1"
        self.cumulative_reward: float = 0.0
        self.box_capacity: float = 10.0
        self.step_count: int = 0
        self.invalid_actions_count: int = 0

global_state = EnvState()

app = FastAPI(title="Warehouse Packing Optimization Environment")

TASKS = {
    "task_1": {
        "items": [{"id": "item_1", "size": 3.0}, {"id": "item_2", "size": 4.0}, {"id": "item_3", "size": 2.0}],
        "box_capacity": 10.0
    },
    "task_2": {
        "items": [
            {"id": "item_1", "size": 5.0}, {"id": "item_2", "size": 4.0}, 
            {"id": "item_3", "size": 3.0}, {"id": "item_4", "size": 2.0}, 
            {"id": "item_5", "size": 6.0}
        ],
        "box_capacity": 10.0
    },
    "task_3": {
        "items": [
            {"id": "item_1", "size": 6.0}, {"id": "item_2", "size": 4.0}, 
            {"id": "item_3", "size": 7.0}, {"id": "item_4", "size": 3.0}, 
            {"id": "item_5", "size": 5.0}, {"id": "item_6", "size": 2.0}, 
            {"id": "item_7", "size": 3.0}, {"id": "item_8", "size": 2.0}
        ],
        "box_capacity": 10.0
    }
}

@app.post("/reset", response_model=StepResult)
async def reset(body: Optional[Dict[str, Any]] = None):
    # Retrieve task
    task_name = "task_1"
    if body and "task_name" in body:
        task_name = body["task_name"]
    elif body and "task" in body:
        task_name = body["task"]
        
    if task_name not in TASKS:
        task_name = "task_1"

    global_state.task_name = task_name
    task_data = TASKS[task_name]
    
    global_state.box_capacity = task_data["box_capacity"]
    global_state.remaining_items = [Item(**i) for i in task_data["items"]]
    global_state.boxes = [Box(id="box_1", capacity=global_state.box_capacity, used_space=0.0)]
    global_state.cumulative_reward = 0.0
    global_state.step_count = 0
    global_state.invalid_actions_count = 0

    obs = Observation(remaining_items=global_state.remaining_items, boxes=global_state.boxes)
    
    return StepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={"message": "Environment reset", "task": task_name}
    )

def calculate_grader_score() -> float:
    total_space = sum(b.capacity for b in global_state.boxes)
    used_space = sum(b.used_space for b in global_state.boxes)
    if total_space == 0:
        return 0.0
    # Normalize score between 0 and 1
    return min(1.0, max(0.0, used_space / total_space))

@app.post("/step", response_model=StepResult)
async def step(action: Action):
    global_state.step_count += 1
    reward = -0.01
    reason = "Step taken. "
    is_invalid = False

    if action.action_type == "open_new_box":
        # Determine if unnecessary
        unnecessary = False
        if len(global_state.remaining_items) > 0:
            min_item_size = min(i.size for i in global_state.remaining_items)
            for b in global_state.boxes:
                if (b.capacity - b.used_space) >= min_item_size:
                    unnecessary = True
                    break
        else:
            unnecessary = True

        if unnecessary:
            reward -= 0.2
            reason += "Unnecessarily opened new box."
        else:
            reward += 0.0
            reason += "Opened new box."

        new_box = Box(id=f"box_{len(global_state.boxes)+1}", capacity=global_state.box_capacity, used_space=0.0)
        global_state.boxes.append(new_box)
    
    elif action.action_type == "place_item":
        if not action.item_id or not action.box_id:
            reward -= 0.3
            reason += "Invalid placement: Must provide item_id and box_id."
            is_invalid = True
        else:
            item_idx = next((i for i, itm in enumerate(global_state.remaining_items) if itm.id == action.item_id), None)
            box_idx = next((i for i, bx in enumerate(global_state.boxes) if bx.id == action.box_id), None)

            if item_idx is None or box_idx is None:
                reward -= 0.3
                reason += "Invalid placement: Item or Box not found."
                is_invalid = True
            else:
                item = global_state.remaining_items[item_idx]
                box = global_state.boxes[box_idx]

                if box.used_space + item.size > box.capacity:
                    reward -= 0.3
                    reason += "Invalid placement: Exceeds box capacity."
                    is_invalid = True
                else:
                    box.used_space += item.size
                    global_state.remaining_items.pop(item_idx)
                    reward += 0.2
                    reason += "Valid placement."

                    utilization = box.used_space / box.capacity
                    if utilization > 0.8:
                        reward += 0.3
                        reason += " High utilization bonus (>80%)."

    if is_invalid:
        global_state.invalid_actions_count += 1

    global_state.cumulative_reward += reward
    done = len(global_state.remaining_items) == 0 or global_state.step_count >= 20

    if done:
        score = calculate_grader_score()
        if score >= 0.8:
            reward += 1.0
            global_state.cumulative_reward += 1.0
            reason += " Bonus: All items packed efficiently."
        
    obs = Observation(remaining_items=global_state.remaining_items, boxes=global_state.boxes)
    
    info = {
        "reason": reason,
        "cumulative_reward": round(global_state.cumulative_reward, 2),
        "grader_score": round(calculate_grader_score(), 3) if done else 0.0,
        "invalid_actions_count": global_state.invalid_actions_count,
        "invalid_action_this_step": is_invalid
    }

    return StepResult(
        observation=obs,
        reward=round(reward, 2),
        done=done,
        info=info
    )

@app.get("/", response_class=HTMLResponse)
async def ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Warehouse Packing Environment</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                color: #333;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .header h1 {
                margin: 0 0 10px 0;
                color: #2c3e50;
            }
            .header p {
                margin: 0;
                color: #7f8c8d;
                font-size: 1.1em;
            }
            .card {
                background: #fff;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }
            .controls {
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                transition: background-color 0.2s;
            }
            button:hover {
                background-color: #2980b9;
            }
            .btn-reset {
                background-color: #e74c3c;
            }
            .btn-reset:hover {
                background-color: #c0392b;
            }
            input[type="text"] {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                min-width: 250px;
                font-family: monospace;
            }
            .metrics {
                display: flex;
                justify-content: space-between;
                gap: 15px;
                flex-wrap: wrap;
            }
            .metric-box {
                flex: 1;
                background: #f1f2f6;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                min-width: 120px;
            }
            .metric-box h3 {
                margin: 0 0 5px 0;
                font-size: 0.9em;
                color: #7f8c8d;
                text-transform: uppercase;
            }
            .metric-box p {
                margin: 0;
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            @media (max-width: 600px) {
                .grid-2 { grid-template-columns: 1fr; }
            }
            .box-list, .item-list {
                display: flex;
                flex-direction: column;
                gap: 10px;
                max-height: 300px;
                overflow-y: auto;
                padding-right: 5px;
            }
            .box-item, .item-item {
                background: #f8f9fa;
                border: 1px solid #eee;
                padding: 10px;
                border-radius: 5px;
            }
            .progress-bar {
                width: 100%;
                background-color: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
                height: 10px;
                margin-top: 8px;
            }
            .progress-fill {
                height: 100%;
                background-color: #2ecc71;
                transition: width 0.3s ease;
            }
            pre {
                background: #2d3436;
                color: #f1f2f6;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                max-height: 300px;
                font-size: 0.9em;
                margin: 0;
            }
            h2 { margin-top: 0; font-size: 1.2em; color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 10px;}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Warehouse Packing Environment</h1>
                <p>Optimize packing efficiency</p>
            </div>

            <div class="card">
                <h2>Controls</h2>
                <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 0;">Tip: Place items before opening new boxes</p>
                <div class="controls" style="margin-bottom: 10px;">
                    <button class="btn-reset" onclick="resetEnv()">Reset</button>
                    <button onclick="presetPlaceFirst()">Place First Item in First Box</button>
                    <button onclick="presetOpenBox()">Open New Box</button>
                    <button style="background-color: #f39c12;" onclick="suggestAction()">Suggest Best Action</button>
                    <button style="background-color: #9b59b6;" onclick="runAutoAgent()">Run Auto Agent (5 steps)</button>
                </div>
                <div class="controls">
                    <input type="text" id="actionInput" placeholder='Enter / Suggest JSON here (e.g. {"action_type": "open_new_box"})'>
                    <button onclick="stepEnv()">Execute Step</button>
                </div>
                <div id="invalid-warning" style="display: none; color: #e74c3c; margin-top: 10px; font-weight: bold; font-size: 0.9em;">
                    Warning: Invalid action executed in previous step!
                </div>
            </div>

            <div class="card">
                <h2>Metrics</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Efficiency Score</h3>
                        <p id="metric-score">0</p>
                    </div>
                    <div class="metric-box">
                        <h3>Boxes Used</h3>
                        <p id="metric-boxes">0</p>
                    </div>
                    <div class="metric-box">
                        <h3>Steps Taken</h3>
                        <p id="metric-steps">0</p>
                    </div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h2>Boxes</h2>
                    <div class="box-list" id="boxes-container">
                        <p style="color: #7f8c8d; font-size: 0.9em;">No boxes open.</p>
                    </div>
                </div>
                <div class="card">
                    <h2>Remaining Items</h2>
                    <div class="item-list" id="items-container">
                        <p style="color: #7f8c8d; font-size: 0.9em;">No items.</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Output Log</h2>
                <pre id="output">No output yet.</pre>
            </div>
        </div>

        <script>
            let localStepCount = 0;
            let currentObservation = null;

            async function updateUI(res) {
                try {
                    const data = await res.json();
                    document.getElementById('output').textContent = JSON.stringify(data, null, 2);

                    if (data.info) {
                        const score = data.info.grader_score !== undefined ? data.info.grader_score.toFixed(2) : '0.00';
                        document.getElementById('metric-score').textContent = score;
                        
                        const warnDisplay = data.info.invalid_action_this_step ? 'block' : 'none';
                        document.getElementById('invalid-warning').style.display = warnDisplay;
                    }

                    if (data.observation) {
                        currentObservation = data.observation;
                        const boxes = data.observation.boxes || [];
                        const items = data.observation.remaining_items || [];

                        document.getElementById('metric-boxes').textContent = boxes.length;
                        document.getElementById('metric-steps').textContent = localStepCount;

                        // Render Boxes
                        const boxesContainer = document.getElementById('boxes-container');
                        boxesContainer.innerHTML = '';
                        if (boxes.length === 0) {
                            boxesContainer.innerHTML = '<p style="color: #7f8c8d; font-size: 0.9em;">No boxes open.</p>';
                        } else {
                            boxes.forEach(box => {
                                const pct = (box.used_space / box.capacity) * 100;
                                const el = document.createElement('div');
                                el.className = 'box-item';
                                el.innerHTML = `
                                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                                        <strong>${box.id}</strong>
                                        <span>${box.used_space.toFixed(1)} / ${box.capacity.toFixed(1)}</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${Math.min(pct, 100)}%;"></div>
                                    </div>
                                `;
                                boxesContainer.appendChild(el);
                            });
                        }

                        // Render Items
                        const itemsContainer = document.getElementById('items-container');
                        itemsContainer.innerHTML = '';
                        if (items.length === 0) {
                            itemsContainer.innerHTML = '<p style="color: #7f8c8d; font-size: 0.9em;">No items left.</p>';
                        } else {
                            items.forEach(item => {
                                const el = document.createElement('div');
                                el.className = 'item-item';
                                el.innerHTML = `
                                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                                        <strong>${item.id}</strong>
                                        <span>Size: ${item.size.toFixed(1)}</span>
                                    </div>
                                `;
                                itemsContainer.appendChild(el);
                            });
                        }
                    }
                } catch(e) {
                    document.getElementById('output').textContent = 'UI rendering error: ' + e;
                }
            }

            async function resetEnv() {
                try {
                    const res = await fetch('/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ task_name: 'task_1' })
                    });
                    localStepCount = 0;
                    document.getElementById('actionInput').value = '';
                    await updateUI(res);
                } catch(e) {
                    document.getElementById('output').textContent = 'Network Error: ' + e;
                }
            }

            function presetPlaceFirst() {
                if (!currentObservation || !currentObservation.remaining_items.length || !currentObservation.boxes.length) {
                    alert("No items or no boxes available.");
                    return;
                }
                const action = {
                    action_type: "place_item",
                    item_id: currentObservation.remaining_items[0].id,
                    box_id: currentObservation.boxes[0].id
                };
                document.getElementById('actionInput').value = JSON.stringify(action);
            }

            function presetOpenBox() {
                const action = { action_type: "open_new_box" };
                document.getElementById('actionInput').value = JSON.stringify(action);
            }

            function getSuggestion() {
                if (!currentObservation || currentObservation.remaining_items.length === 0) {
                    return null;
                }
                const firstItem = currentObservation.remaining_items[0];
                let targetBox = null;
                for (let box of currentObservation.boxes) {
                    if (box.capacity - box.used_space >= firstItem.size) {
                        targetBox = box;
                        break;
                    }
                }
                if (targetBox) {
                    return {
                        action_type: "place_item",
                        item_id: firstItem.id,
                        box_id: targetBox.id
                    };
                } else {
                    return { action_type: "open_new_box" };
                }
            }

            function suggestAction() {
                const action = getSuggestion();
                if (action) {
                    document.getElementById('actionInput').value = JSON.stringify(action);
                } else {
                    alert("No remaining items to pack!");
                }
            }

            async function stepEnv(overrideActionObj = null) {
                let actionObj = overrideActionObj;
                if (!actionObj) {
                    let actionText = document.getElementById('actionInput').value.trim();
                    if (!actionText) {
                        alert("Please enter a valid action JSON");
                        return;
                    }
                    try {
                        actionObj = JSON.parse(actionText);
                    } catch(e) {
                        alert("Invalid JSON in custom action input.");
                        return;
                    }
                }

                try {
                    const res = await fetch('/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(actionObj)
                    });
                    localStepCount++;
                    await updateUI(res);
                } catch(e) {
                    document.getElementById('output').textContent = 'Network Error: ' + e;
                }
            }

            async function runAutoAgent() {
                for (let i = 0; i < 5; i++) {
                    const suggestion = getSuggestion();
                    if (!suggestion) break;
                    
                    document.getElementById('actionInput').value = JSON.stringify(suggestion);
                    await stepEnv(suggestion);
                    
                    await new Promise(r => setTimeout(r, 400));
                }
            }

            // Init load
            window.onload = resetEnv;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/state")
async def state():
    return {
        "remaining_items": [i.model_dump() for i in global_state.remaining_items],
        "boxes": [b.model_dump() for b in global_state.boxes],
        "task_name": global_state.task_name,
        "cumulative_reward": round(global_state.cumulative_reward, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
