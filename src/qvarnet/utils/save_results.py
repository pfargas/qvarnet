import os


def save_results(base_path, **kwargs) -> bool:
    try:
        os.makedirs(base_path, exist_ok=True)
        # Save other results as needed
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

    for key, value in kwargs.items():
        try:
            # if it's a dictionary, save as a json
            if isinstance(value, dict):
                import json

                with open(os.path.join(base_path, f"{key}.json"), "w") as f:
                    json.dump(value, f, indent=4)
                continue
            with open(os.path.join(base_path, f"{key}.txt"), "w") as f:
                for num in value:
                    f.write(str(num))
                    f.write("\n")
        except Exception as e:
            print(f"Error saving {key}: {e}")
            return False
    return True
