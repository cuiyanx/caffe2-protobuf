var protobuf = require("protobufjs");
var caffe2 = require("./caffe2.js");
var fs = require('fs');
var init_path = "./models/init_net_int8.pb";
var predict_path = "./models/predict_net_int8.pb";

var buffer = fs.readFileSync(predict_path);
var message = caffe2.caffe2.NetDef.decode(buffer);
console.log(message.op[0]);
