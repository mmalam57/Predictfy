from backend_predictify import app
from backend_predictify import functions
from flask import jsonify

@app.route('/',methods=['GET'])
def index():
    res = functions.testing()
    res_json = float(res)
    return jsonify(res_json)