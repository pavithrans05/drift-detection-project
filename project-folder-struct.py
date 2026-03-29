import os

def print_tree_limited(start_path, max_depth=2, current_depth=0, indent=""):
    if current_depth > max_depth:
        return

    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        print(indent + "Permission Denied")
        return

    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(indent + connector + item)

        if os.path.isdir(path) and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            print_tree_limited(path, max_depth, current_depth + 1, indent + extension)

# ✅ Your project path
project_path = r"D:\drift-detection_honours-project"

print(project_path)
print_tree_limited(project_path, max_depth=2)