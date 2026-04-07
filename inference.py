import os
import json
import time
import subprocess
import requests
from openai import OpenAI

def from_docker_image(image_name: str, port: int = 7860):
    class LocalEnv:
        def __init__(self, image: str, p: int):
            self.image = image
            self.port = p
            self.container_id = None
            self.url = f"http://localhost:{self.port}"
        
        def start(self):
            print(f"Starting docker container for {self.image} on port {self.port}...")
            cmd = ["docker", "run", "-d", "-p", f"{self.port}:7860", self.image]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to start container: {result.stderr}")
            self.container_id = result.stdout.strip()
            
            # Wait for server health
            for _ in range(30):
                try:
                    resp = requests.post(f"{self.url}/reset", json={"task_name":"task_1"})
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                time.sleep(1)
            raise Exception("Timeout waiting for container server to start")

        def stop(self):
            if self.container_id:
                # Stop and rm without breaking if already stopped
                subprocess.run(["docker", "stop", self.container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["docker", "rm", self.container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def reset(self, task_name="task_1"):
            res = requests.post(f"{self.url}/reset", json={"task_name": task_name})
            res.raise_for_status()
            return res.json()

        def step(self, action):
            res = requests.post(f"{self.url}/step", json=action)
            res.raise_for_status()
            return res.json()
            
    env = LocalEnv(image_name, port)
    env.start()
    return env

def format_bool(val: bool) -> str:
    return "true" if val else "false"

def main():
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    local_image_name = os.environ.get("LOCAL_IMAGE_NAME", "warehouse-env")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    env = from_docker_image(local_image_name)
    tasks = ["task_1", "task_2", "task_3"]

    system_prompt = (
        "You are an AI packing warehouse items into boxes to optimize space.\n"
        "Available actions: \n"
        "1. {\"action_type\": \"place_item\", \"item_id\": \"item_X\", \"box_id\": \"box_Y\"}\n"
        "2. {\"action_type\": \"open_new_box\"}\n"
        "Output ONLY a JSON dict with your chosen action."
    )

    try:
        for task_name in tasks:
            print(f"[START] task={task_name} env=warehouse-env model={model_name}")
            try:
                step_res = env.reset(task_name=task_name)
                done = step_res.get("done", False)
                step_count = 0
                rewards = []
                score = 0.0
                success = False
                
                while not done and step_count < 30:
                    step_count += 1
                    obs = step_res["observation"]
                    user_msg = f"Observation: {json.dumps(obs)}"
                    
                    action_dict = None
                    error_msg = "null"
                    
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_msg}
                            ],
                            temperature=0.0
                        )
                        action_str = response.choices[0].message.content.strip()
                        
                        # Extract dict from potential markdown
                        if "{" in action_str and "}" in action_str:
                            start = action_str.find("{")
                            end = action_str.rfind("}") + 1
                            action_dict = json.loads(action_str[start:end])
                        else:
                            error_msg = "Failed to parse action json"
                            action_dict = {"action_type": "open_new_box"}
                    except Exception as e:
                        error_msg = str(e).replace('\n', ' ')
                        action_dict = {"action_type": "open_new_box"}
                    
                    try:
                        step_res = env.step(action_dict)
                        reward = float(step_res.get("reward", 0.0))
                        done = step_res.get("done", False)
                        rewards.append(reward)
                        
                        print(f"[STEP] step={step_count} action={json.dumps(action_dict)} reward={reward:.2f} done={format_bool(done)} error=\"{error_msg}\"")
                        
                        if done:
                            score = float(step_res.get("info", {}).get("grader_score", 0.0))
                            success = score >= 0.8
                    except Exception as e:
                        err_str = str(e).replace('\n', ' ')
                        print(f"[STEP] step={step_count} action={json.dumps(action_dict)} reward=0.00 done=false error=\"{err_str}\"")
                        break

                rewards_str = ",".join([f"{r:.2f}" for r in rewards])
                print(f"[END] success={format_bool(success)} steps={step_count} score={score:.3f} rewards={rewards_str}")
            except Exception as e:
                print(f"[END] success=false steps=0 score=0.000 rewards=")
                print(f"Error executing task {task_name}: {e}")
    finally:
        env.stop()

if __name__ == "__main__":
    main()
