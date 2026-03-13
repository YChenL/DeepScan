import os
import inspect
from .policy import QuestionSample


policy_map = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
py_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and not f.startswith('__')]
for file in py_files:
    module_name = file[:-3]
    if module_name != 'base':  
        exec(f"from .{module_name} import *")

for name, obj in list(locals().items()):
    if (inspect.isclass(obj) and 
        issubclass(obj, QuestionSample) and 
        obj != QuestionSample):
        policy_name = name.replace('QuestionSample', '').lower()
        policy_map[policy_name] = obj
