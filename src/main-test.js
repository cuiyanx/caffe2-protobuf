var protobuf = require("protobufjs");
var caffe2 = require("./caffe2-local.js");
var fs = require('fs');
var Caffe2ModelUtils = require("./Caffe2ModelUtils.js");

var init_path = "./models/init_net_int8.pb";
var predict_path = "./models/predict_net_int8.pb";
var JSONpath = "./output/caffe2-net.json";

var init_buffer = fs.readFileSync(init_path);
var predict_buffer = fs.readFileSync(predict_path);
var init_message = caffe2.caffe2.NetDef.decode(init_buffer);
var predict_message = caffe2.caffe2.NetDef.decode(predict_buffer);

var util = new Caffe2ModelUtils(predict_message, init_message, true);
console.log(util.getCaffe2Model()[28].arg);
