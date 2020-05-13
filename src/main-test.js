var protobuf = require("protobufjs");
var caffe2 = require("./caffe2-local.js");
var fs = require('fs');
var Caffe2ModelImporter = require("./Caffe2ModelImporter.js");
var Caffe2ModelUtils = require("./Caffe2ModelUtils.js");

var init_path = "./models/init_net_int8.pb";
var predict_path = "./models/predict_net_int8.pb";
var JSONpath = "./output/caffe2-net.json";

var init_buffer = fs.readFileSync(init_path);
var predict_buffer = fs.readFileSync(predict_path);
var init_message = caffe2.caffe2.NetDef.decode(init_buffer);
var predict_message = caffe2.caffe2.NetDef.decode(predict_buffer);

var configs = {
  rawModel: predict_message,
  weightModel: init_message,
  backend: "WebML",
  prefer: "fast",
  softmax: false,
  inputSize: [1, 226, 226, 3],
  outputSize: [1, 1, 1000, 1],
  isQuantized: true
};

var Caffe2ModelImporter = new Caffe2ModelImporter(configs);
Caffe2ModelImporter.createCompiledModel();
