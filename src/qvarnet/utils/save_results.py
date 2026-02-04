import os


def save_results(base_path, csv_delimiters, **kwargs) -> bool:
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
            elif (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
                and all(
                    any(delim in item for delim in csv_delimiters) for item in value
                )
            ):
                with open(os.path.join(base_path, f"{key}.csv"), "w") as f:
                    for line in value:
                        f.write(line)
                        f.write("\n")
                continue
            # otherwise, save as a text file
            with open(os.path.join(base_path, f"{key}.txt"), "w") as f:
                for num in value:
                    f.write(str(num))
                    f.write("\n")
        except Exception as e:
            print(f"Error saving {key}: {e}")
            return False
    return True
