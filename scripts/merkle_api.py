import json

from flask import Flask, jsonify, request


app = Flask(__name__)

with open("scripts/api_data.json", "r", encoding="utf-8") as data_file:
    api_data = json.load(data_file)


@app.route("/merkle", methods=["GET"])
def merkle():
    if request.method == "GET":
        return jsonify(api_data)


if __name__ == "__main__":
    app.run(debug=True)
