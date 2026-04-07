# Warehouse Packing Optimization Environment

This is an OpenEnv-compliant environment simulating a warehouse worker packing items into boxes efficiently to minimize wasted space and the number of boxes used.

## 1. Environment Description
The environment challenges an agent to pack a given set of items (represented by their 1D sizes) into uniformly sized boxes. The goal is to maximize efficiency, represented by the ratio of used space to total allocated space, while following strict box capacities.

## 2. Observation Space
The state returned by the environment consists of:
- `remaining_items`: A list of items yet to be packed, each containing an `id` and `size`.
- `boxes`: A list of currently open boxes, each containing an `id`, `capacity`, and `used_space`.

## 3. Action Space
The agent can execute the following actions formatted as JSON:
- `{"action_type": "place_item", "item_id": "...", "box_id": "..."}`: Places a specific item into the designated open box.
- `{"action_type": "open_new_box"}`: Opens a new empty box, increasing total packing capacity (but temporarily lowering utility).

## 4. Task Descriptions
- **Task 1 (Easy)**: Single box packing. Involves packing a few items that fit perfectly within a single box's capacity.
- **Task 2 (Medium)**: Multiple boxes. Requires opening a second box to accommodate more items efficiently.
- **Task 3 (Hard)**: Optimize packing efficiency. Given a large set of items and fixed box capacity, finding the optimal arrangement to leave minimum wasted space across multiple boxes.

## 5. Setup Instructions
To build and run this environment manually:

```bash
# Build the Docker image
docker build -t warehouse-env .

# Run the server manually
docker run -p 7860:7860 warehouse-env
```

To run the inference workflow directly using OpenAI-compatible APIs:
```bash
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your-api-key"
export LOCAL_IMAGE_NAME="warehouse-env"
python inference.py
```

## 6. Baseline Scores
- Task 1: 0.900 efficiency score.
- Task 2: ~0.800 efficiency score.
- Task 3: ~0.950 efficiency score or higher depending on the LLM's automated packing algorithm.
