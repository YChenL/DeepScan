import os
import inspect
from .policy import QuestionSample

# Automatically import all Python files in current directory
policy_map = {}
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get all Python files in current directory
py_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and not f.startswith('__')]

# Dynamically import all modules
for file in py_files:
    module_name = file[:-3]  # Remove .py suffix
    if module_name != 'policy':  # Skip base class file
        # Use from syntax for direct import
        exec(f"from .{module_name} import *")

# Iterate through all members of current module
for name, obj in list(locals().items()):
    # Check if it's a subclass of QuestionSample
    if (inspect.isclass(obj) and 
        issubclass(obj, QuestionSample) and 
        obj != QuestionSample):
        # Convert class name to policy name (remove QuestionSample suffix, lowercase)
        policy_name = name.replace('QuestionSample', '').lower()
        policy_map[policy_name] = obj
