from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("template"))
template = env.get_template("index.html")

print("Template loaded successfully")