import json
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import werkzeug
import cv2
import numpy as np
from triip_detect import detect
import os
from dotenv import load_dotenv
from flask_cors import CORS

from s3_controller import s3_upload
from db_controller import importData, readPendingReq, readAllReq, readRejectedReq, readApprovedReq

app = Flask(__name__)
CORS(app)
api = Api(app)

load_dotenv()
ACCESS_POINT = os.getenv('ACCESS_POINT')

class Home(Resource):
    def get(self):
        try:
            return {'server_status': 'OK'}, 200
        except Exception as ex:
            return {'msg': "Something's happened",
                    'error details': ex}, 500

class send_kyc_request(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser(bundle_errors=True)
            parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', required=True)
            parser.add_argument('user_id', location='form', required=True)
            parser.add_argument('username', location='form', required=True)
            parser.add_argument('address', location='form', required=True)
            parser.add_argument('phone', location='form', required=True)
            args = parser.parse_args()
            uploaded_image = args['file'].read()
            img = cv2.imdecode(np.frombuffer(uploaded_image, np.uint8), cv2.IMREAD_COLOR)
            result = detect(img)
            detectedElements = []
            for key in result:
                p = result[key]
                img = cv2.rectangle(img, (p[0], p[1]) , (p[2], p[3]), (255, 0, 0), 2)
                detectedElements.append(key)
            s3_upload(img, args['user_id'] + '.png')
            importData(args['user_id'], args['username'], args['address'], args['phone'], ACCESS_POINT + args['user_id'] + '.png', detectedElements)
            return {
                'Status': 'Success!'
            }
        except Exception as ex:
            print(ex)
            return {
                'msg': "Something's happened",
                'description': str(ex)
                }, 500

class getAllRequest(Resource):
    def get(self):
        return jsonify(readAllReq())

class getRejectedRequest(Resource):
    def get(self):
        return jsonify(readRejectedReq())

class getPendingRequest(Resource):
    def get(self):
        return jsonify(readPendingReq())

class getApprovedRequest(Resource):
    def get(self):
        return jsonify(readApprovedReq())

#endpoint(s)
api.add_resource(Home, "/")
api.add_resource(send_kyc_request, "/kyc")
api.add_resource(getAllRequest, "/get/all")
api.add_resource(getRejectedRequest, "/get/rejected")
api.add_resource(getPendingRequest, "/get/pending")
api.add_resource(getApprovedRequest, "/get/approved")

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5050)