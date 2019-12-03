from flask import Flask
from flask_restful import Api
from similarity import SimilarityMeasurer

app = Flask(__name__)
api = Api(app)

api.add_resource(SimilarityMeasurer, '/api/similarity')


if __name__ == '__main__':
    app.run(port=8000, debug=False)