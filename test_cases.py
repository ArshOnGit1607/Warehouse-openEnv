from fastapi.testclient import TestClient
from main import app, global_state

client = TestClient(app)

def run_tests():
    print("--- Starting Edge Case Tests ---")
    
    # Test 0: Reset the environment
    res = client.post("/reset", json={"task_name": "task_1"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["observation"]["remaining_items"]) == 3
    assert len(data["observation"]["boxes"]) == 1
    print("✅ Reset works.")

    # Test 1: Invalid placement (Item doesn't exist)
    res = client.post("/step", json={"action_type": "place_item", "item_id": "non_existent", "box_id": "box_1"})
    data = res.json()
    assert "Invalid placement" in data["info"]["reason"]
    assert data["info"]["invalid_action_this_step"] == True
    print("✅ Item doesn't exist handled.")

    # Test 2: Invalid placement (Box doesn't exist)
    res = client.post("/step", json={"action_type": "place_item", "item_id": "item_1", "box_id": "non_existent"})
    data = res.json()
    assert "Invalid placement" in data["info"]["reason"]
    assert data["info"]["invalid_action_this_step"] == True
    print("✅ Box doesn't exist handled.")

    # Test 3: Open an unnecessary box
    res = client.post("/step", json={"action_type": "open_new_box"})
    data = res.json()
    assert "Unnecessarily opened" in data["info"]["reason"]
    assert len(data["observation"]["boxes"]) == 2
    print("✅ Unnecessary box tracked and punished.")

    # Test 4: Valid placement and capacity reduction
    # item_1 size is 3.0, capacity is 10.0
    res = client.post("/step", json={"action_type": "place_item", "item_id": "item_1", "box_id": "box_1"})
    data = res.json()
    assert "Valid placement" in data["info"]["reason"]
    box_1 = next(b for b in data["observation"]["boxes"] if b["id"] == "box_1")
    assert box_1["used_space"] == 3.0
    remaining_items = data["observation"]["remaining_items"]
    assert len(remaining_items) == 2
    print("✅ Valid placement works.")
    
    # Test 5: Exceeding Box Capacity (box_1 now has 7.0 left)
    # We will try to add items exceeding 7.0
    # Add a huge item to trigger it by directly modifying state (not possible), so let's try opening items or just adding size to it.
    # Actually wait, let's open task_2 to get bigger items
    client.post("/reset", json={"task_name": "task_2"})
    # task_2 items: 5.0, 4.0, 3.0, 2.0, 6.0
    # place 5.0 and 6.0 -> 11.0 > 10.0
    client.post("/step", json={"action_type": "place_item", "item_id": "item_1", "box_id": "box_1"}) # 5.0
    res = client.post("/step", json={"action_type": "place_item", "item_id": "item_5", "box_id": "box_1"}) # 6.0
    data = res.json()
    assert "Exceeds box capacity" in data["info"]["reason"]
    box_1 = next(b for b in data["observation"]["boxes"] if b["id"] == "box_1")
    assert box_1["used_space"] == 5.0 # Shouldn't have changed
    print("✅ Capacity limit enforced.")

    # Test 6: Missing keys handling by FastAPI
    # This should return 422 Unprocessable Entity
    res = client.post("/step", json={"action_type": "place_item"})
    if res.status_code == 200:
        data = res.json()
        assert data["info"]["invalid_action_this_step"] == True
    else:
        assert res.status_code in [400, 422]
    print("✅ Missing keys validation handled.")

    # Test 7: Max step threshold
    client.post("/reset", json={"task_name": "task_1"})
    for i in range(20):
        res = client.post("/step", json={"action_type": "open_new_box"})
    data = res.json()
    assert data["done"] == True
    print("✅ Max steps threshold (20) triggers 'done'.")

    # Test 8: Success condition triggers 'done'
    client.post("/reset", json={"task_name": "task_1"})
    # item_1 (3), item_2 (4), item_3 (2) = 9.0 total < 10.0
    client.post("/step", json={"action_type": "place_item", "item_id": "item_1", "box_id": "box_1"})
    client.post("/step", json={"action_type": "place_item", "item_id": "item_2", "box_id": "box_1"})
    res = client.post("/step", json={"action_type": "place_item", "item_id": "item_3", "box_id": "box_1"})
    data = res.json()
    assert data["done"] == True
    assert "Bonus:" in data["info"]["reason"]
    print("✅ Task completion handled.")
    
    print("--- All Tests Passed Successfully ---")

if __name__ == "__main__":
    run_tests()
